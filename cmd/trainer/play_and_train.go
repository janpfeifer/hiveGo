package main

// This file implements continuous play and train.

import (
	"context"
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"slices"
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

type MatchStats struct {
	mu          sync.Mutex
	TotalCount  int
	lastMatches []*Match
	Wins        [2]int
	Draws       int
}

const NumMatchesToKeepForStats = 200

func NewMatchStats() *MatchStats {
	return &MatchStats{
		lastMatches: make([]*Match, 0, NumMatchesToKeepForStats),
	}
}

func (ms *MatchStats) AddResult(match *Match) {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if len(ms.lastMatches) >= NumMatchesToKeepForStats {
		// Remove counts for oldest match.
		oldMatch := ms.lastMatches[ms.TotalCount%NumMatchesToKeepForStats]
		b := oldMatch.FinalBoard()
		if b.Draw() {
			ms.Draws--
		} else {
			player := b.Winner()
			if oldMatch.Swapped {
				player = 1 - player
			}
			ms.Wins[player]--
		}
		// Replace with new match.
		ms.lastMatches[ms.TotalCount%NumMatchesToKeepForStats] = match
	} else {
		// Add new match.
		ms.lastMatches = append(ms.lastMatches, match)
	}
	ms.TotalCount++

	// Add count of this match.
	b := match.FinalBoard()
	if b.Draw() {
		ms.Draws++
	} else {
		player := b.Winner()
		if match.Swapped {
			player = 1 - player
		}
		ms.Wins[player]++
	}
}

// Players[0] is the one being trained, and also playing.
// If Players[1] is the same as players[0], it is also
// used for training, otherwise their moves are discarded.
//
// If interrupted return nil. If something failed (like saving model), returns the error.
func playAndTrain(ctx context.Context) error {
	parallelism := getParallelism()
	if isSamePlayer {
		klog.Infof("Same player playing both sides, learning from both.")
	}

	// Generate played games.
	matchesChan := make(chan *Match, 5)
	matchIdGen := &IdGen{}
	matchStats := NewMatchStats()
	for i := 0; i < parallelism; i++ {
		go continuouslyPlay(ctx, matchIdGen, matchStats, matchesChan)
	}

	// Rescore player1's moves, for learning.
	if !isSamePlayer {
		rescoreMatchesChan := make(chan *Match, 2*parallelism)
		klog.Infof("Rescoring player 1's moves for learning (parallelism=%d).", parallelism)
		for range parallelism {
			go continuouslyRescoreMatches(ctx, matchesChan, rescoreMatchesChan, PlayerSecond)
		}
		// Swap matchesChan and rescoreMatchesChan, so that matchesChan holds the
		// correctly labeled matches.
		matchesChan = rescoreMatchesChan
	}

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
	klog.Infof("Saving on exit", aiPlayers[0].Scorer)
	return aiPlayers[0].Learner.Save()
}

func continuouslyPlay(ctx context.Context, matchIdGen *IdGen, matchStats *MatchStats, matchesChan chan<- *Match) {
	// Defer clean exit, if interrupted.
	defer func() {
		klog.Infof("Continues playing interrupted: %v", ctx.Err())
		close(matchesChan)
	}()
	for {
		id := matchIdGen.NextId()
		match := runMatch(ctx, id)
		if ctx.Err() != nil {
			klog.Infof("Match %d interrupted: context cancelled with %v", id, ctx.Err())
			return
		}
		if match == nil {
			klog.Errorf("runMatch() returned a nil Match!? Something went wrong.")
			return
		}
		matchStats.AddResult(match)
		msg := "draw"
		if !match.FinalBoard().Draw() {
			player := match.FinalBoard().Winner()
			if match.Swapped {
				player = 1 - player
			}
			msg = fmt.Sprintf("player %d wins", player)
		}
		klog.V(1).Infof("Match %d finished (%s in %d moves). %d matches played so far. "+
			"Last %d results: p0 win=%d, p1 win=%d, draw=%d",
			id, msg, len(match.Actions), matchStats.TotalCount, NumMatchesToKeepForStats,
			matchStats.Wins[0], matchStats.Wins[1], matchStats.Draws)
		select {
		case <-ctx.Done():
			klog.Infof("Match %d interrupted: context cancelled with %v", id, ctx.Err())
			return
		case matchesChan <- match:
			// Done.
		}
	}
}

// continuouslyRescoreMatches should be called concurrently, and it will indefinitely (or until ctx is cancelled)
// read from matchesIn, and rescore moves from the selected players (or both players if none is given).
//
// It automatically handles matches with swapped players (Match.Swapped), in which case the rescorePlayers are interpreted
// in reverse.
func continuouslyRescoreMatches(ctx context.Context, matchesIn <-chan *Match, matchesOut chan<- *Match, rescorePlayers ...PlayerNum) {
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
			player := board.NextPlayer
			if match.Swapped {
				player = 1 - player
			}
			if len(rescorePlayers) != 0 && slices.Index(rescorePlayers, player) == -1 {
				// No need to rescore.
				continue
			}

			// Rescore as player 0:
			_, _, newScore, _ := aiPlayers[0].Play(board)
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
}
