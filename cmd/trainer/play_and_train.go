package main

// This file implements continuous play and train.

import (
	"context"
	"flag"
	"fmt"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"sync"
	"time"
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
	if !isSamePlayer && *flagRescore {
		rescoreMatchesChan := make(chan *Match, 5)
		for i := 0; i < parallelism; i++ {
			go continuouslyRescorePlayer1(ctx, matchesChan, rescoreMatchesChan)
		}
		// Swap matchesChan and rescoreMatchesChan, so that matchesChan holds the
		// correctly labeled matches.
		matchesChan = rescoreMatchesChan
	}

	// Continuously learn.
	return continuousLearning(ctx, matchesChan)
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

// continuouslyRescorePlayer1 should be called concurrently, and it will indefinitely (or until ctx is cancelled)
// read from matchesIn, rescore the player 1 with teh ai of the player 0, and send the updated match to matchesOut.
func continuouslyRescorePlayer1(ctx context.Context, matchesIn <-chan *Match, matchesOut chan<- *Match) {
	defer func() {
		close(matchesOut)
	}()
	for match := range matchesIn {
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
			if player == 0 {
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
		select {
		case <-ctx.Done():
			klog.Errorf("Rescoring player 1 moves interrupted (context cancelled): %v", ctx.Err())
			return
		case matchesOut <- match:
			// Done
		}
	}
}

// continuousMatchesToLabeledExamples takes batches, and send labeled examples for training.
// It generate labeled examples from the latest 2 batches.
func continuousMatchesToLabeledExamples(ctx context.Context, batchChan <-chan []*Match, labeledExamplesChan chan<- LabeledBoards) {
	batchSize := *flagContinuosPlayAndTrainBatchMatches
	var previousBatch []*Match
	for batch := range batchChan {
		klog.V(1).Infof("Batch received.")

		if *flagContinuosPlayAndTrainKeepPreviousBatch {
			// Merge new batch into previous batch.
			if previousBatch == nil {
				klog.V(1).Infof("Storing batch until 2 are available.")
				previousBatch = batch
				continue
			}

			if len(previousBatch) > batchSize {
				tmpBatch := make([]*Match, 0, 2*batchSize)
				tmpBatch = append(tmpBatch,
					previousBatch[len(previousBatch)-batchSize:]...)
				tmpBatch = append(tmpBatch, batch...)
				previousBatch = tmpBatch
			} else {
				previousBatch = append(previousBatch, batch...)
			}
			klog.V(1).Infof("Batch aggregated: generating labeled examples.")
		} else {
			previousBatch = batch
		}

		n := 0
		for _, match := range previousBatch {
			n += len(match.Actions) + 42
		}
		if !isSamePlayer && !*flagDistill && !*flagRescore {
			n /= 2
		}
		le := LabeledBoards{
			Boards:        make([]*Board, 0, n),
			Labels:        make([]float32, 0, n),
			ActionsLabels: make([][]float32, 0, n),
		}
		for _, match := range previousBatch {
			if isSamePlayer || *flagDistill || *flagRescore {
				// Include both players data.
				if *flagLearnWithEndScore {
					labelWithEndScore(match)
				}
				match.AppendToLabeledBoards(&le)
			} else {
				// Only include data from player training.
				if match.Swapped {
					match.AppendToLabeledBoardsForPlayers(&le, [2]bool{false, true})
				} else {
					match.AppendToLabeledBoardsForPlayers(&le, [2]bool{true, false})
				}
			}
		}
		klog.V(1).Infof("Labeled examples issued.")
		labeledExamplesChan <- le
		klog.V(3).Infof("Labels=%v", le.Labels)
	}
}
