package main

// This file implements continuous play and train.

import (
	"context"
	"flag"
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"golang.org/x/sync/errgroup"
	"k8s.io/klog/v2"
	"sync"
)

var (
	flagContinuosPlayAndTrain = flag.Bool("play_and_train",
		false, "If set, continuously play and train matches.")
)

type IdGen struct {
	mu     sync.Mutex
	nextId int
}

func (ig *IdGen) NextId() int {
	ig.mu.Lock()
	defer ig.mu.Unlock()
	id := ig.nextId
	ig.nextId++
	return id
}

type PerPlayerStats struct {
	Wins, Draws, Losses int
}

type MatchStats struct {
	mu          sync.Mutex
	TotalCount  int
	lastMatches []*Match
	PerPlayer   []PerPlayerStats
}

const NumMatchesToKeepForStats = 200

func NewMatchStats() *MatchStats {
	return &MatchStats{
		lastMatches: make([]*Match, 0, NumMatchesToKeepForStats),
		PerPlayer:   make([]PerPlayerStats, 0, len(aiPlayers)),
	}
}

func (ms *MatchStats) AddResult(match *Match) {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if len(ms.lastMatches) >= NumMatchesToKeepForStats {
		// Replace old match.
		ms.lastMatches[ms.TotalCount%NumMatchesToKeepForStats] = match
	} else {
		// Add new match.
		ms.lastMatches = append(ms.lastMatches, match)
	}
	ms.TotalCount++

	// Update totals:
	// TODO: matrix of first player vs second player stats.
	//b := match.FinalBoard()
	//if b.Draw() {
	//	ms.Draws++
	//} else {
	//	player := b.Winner()
	//	if match.Swapped {
	//		player = 1 - player
	//	}
	//	ms.Wins[player]++
	//}
}

// Players[0] is the one being trained, and also playing.
// If Players[1] is the same as players[0], it is also
// used for training, otherwise their moves are discarded.
//
// If interrupted return nil. If something failed (like saving model), returns the error.
func playAndTrain(ctx context.Context) error {
	parallelism := getParallelism()

	// Generate played games.
	matchesChan := make(chan *Match, 5)
	matchIdGen := &IdGen{}
	matchStats := NewMatchStats()
	wgPlay, ctxPlay := errgroup.WithContext(ctx)
	for i := 0; i < parallelism; i++ {
		wgPlay.Go(func() error {
			return continuouslyPlay(ctxPlay, matchIdGen, matchStats, matchesChan)
		})
	}
	go func(c chan *Match) {
		err := wgPlay.Wait()
		if err != nil {
			klog.Errorf("Playing a match failed: %+v", err)
		}
		fmt.Println("Closing matchesChan")
		close(c)
	}(matchesChan)

	// Rescore moves, for learning.
	rescoreMatchesChan := make(chan *Match, 2*parallelism)
	wgRescore, ctxRescore := errgroup.WithContext(ctx)
	klog.Infof("Rescoring matches for learning (parallelism=%d).", parallelism)
	for range parallelism {
		wgRescore.Go(func() error {
			return continuouslyRescoreMatches(ctxRescore, matchesChan, rescoreMatchesChan)
		})
	}
	go func() {
		err := wgPlay.Wait()
		if err != nil {
			klog.Errorf("Rescoring a match failed: %+v", err)
		}
		close(rescoreMatchesChan)
	}()
	matchesChan = rescoreMatchesChan

	// Rescore from end-game score.
	if *flagTrainWithEndScore > 0 {
		klog.Infof("End-game score rescoring with weight %g", *flagTrainWithEndScore)
		rescoreMatchesChan := make(chan *Match, 2*parallelism)
		go continuouslyRescoreWithEndScore(ctx, matchesChan, rescoreMatchesChan)
		matchesChan = rescoreMatchesChan
	}

	// Continuously learn.
	err := continuousLearning(ctx, matchesChan)
	if err != nil {
		return err
	}
	klog.Infof("Saving on exit: %s", aiPlayers[0].Scorer)
	return trainingAI.Learner.Save()
}

func continuouslyPlay(ctx context.Context, matchIdGen *IdGen, matchStats *MatchStats, matchesChan chan<- *Match) error {
	exception := exceptions.Try(func() {
		for {
			id := matchIdGen.NextId()
			match := runMatch(ctx, id, true)
			if ctx.Err() != nil {
				klog.Infof("Match %d interrupted: context cancelled with %v", id, ctx.Err())
				return
			}
			if match == nil {
				klog.Errorf("runMatch() returned a nil Match!? Something went wrong.")
				return
			}
			matchStats.AddResult(match)
			if klog.V(1).Enabled() {
				msg := "draw"
				if !match.FinalBoard().Draw() {
					player := match.FinalBoard().Winner()
					msg = fmt.Sprintf("player %d wins", match.PlayersIdx[player])
				}
				klog.Infof("Match %d finished (%s in %d moves). %d matches played so far. ",
					id, msg, len(match.Actions), matchStats.TotalCount)
			}
			select {
			case <-ctx.Done():
				klog.Infof("Match %d interrupted: context cancelled with %v", id, ctx.Err())
				return
			case matchesChan <- match:
				// Done.
			}
		}
	})
	return exceptionToError(exception)
}

func exceptionToError(exception any) error {
	if exception != nil {
		if err, ok := exception.(error); ok {
			return err
		}
		return errors.Errorf("match failed with exception: %v", exception)
	}
	return nil
}

// continuouslyRescoreMatches should be called concurrently, and it will indefinitely (or until ctx is cancelled)
// read from matchesIn, and rescore moves from the selected players (or both players if none is given).
//
// It automatically handles matches with swapped players (Match.Swapped), in which case the rescorePlayers are interpreted
// in reverse.
func continuouslyRescoreMatches(ctx context.Context, matchesIn <-chan *Match, matchesOut chan<- *Match, rescorePlayers ...PlayerNum) error {
	exception := exceptions.Try(func() {
		for match := range generics.IterChanWithContext(ctx, matchesIn) {
			klog.V(1).Infof("Match received for rescoring.")
			// Pick only moves done by player 0 (or both if they are the same)
			from, to, _ := match.SelectRangeOfActions()
			for idx := range match.Actions {
				if idx < from || idx >= to {
					// We only need to rescore those actions that are actually going
					// to be used.
					continue
				}
				board := match.Boards[idx]
				if board.NumActions() < 2 {
					// Skip when there is none or only one action available.
					continue
				}

				// Rescore as player 0:
				_, _, newScore, _ := trainingAI.Play(board)
				//newScores, ActionsLabels := aiPlayers[0].Searcher.ScoreMatch(
				//	board, match.Actions[idx:idx+1])

				// Clone over new scores and labels for the particular action on the match.
				match.Scores[idx] = newScore
				//if ActionsLabels != nil {
				//	match.ActionsLabels[idx] = ActionsLabels[0]
				//}
			}
			klog.V(1).Infof("Rescored match issued.")
			if !generics.WriteToChanWithContext(ctx, matchesOut, match) {
				// Context cancelled, we are done.
				return
			}
		}
		return
	})
	return exceptionToError(exception)
}
