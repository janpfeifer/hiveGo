package main

// This file implements continuous play and train.

import (
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
	flagContinuosPlayAndTrainKeepPreviousBatch = flag.Bool("play_and_train_keep_previous_batch",
		false, "If set, keep batch of matches for two learning cycles.")
	flagContinuosPlayAndTrainBatchMatches = flag.Int(
		"play_and_train_batch_matches",
		10, "Number of matches to batch before learning.")

	// flagBatchSize should be moved to the learner.
	flagBatchSize = flag.Int("batch_size", 32, "Batch size to use in training.")
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
		// Append new match.
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
func playAndTrain() {
	parallelism := getParallelism()
	setAutoBatchSizesForParallelism(parallelism)
	if isSamePlayer {
		klog.Infof("Same player playing both sides, learning from both.")
	}

	// Generate played games.
	matchChan := make(chan *Match, 5)
	matchIdGen := &IdGen{}
	matchStats := NewMatchStats()
	for i := 0; i < parallelism; i++ {
		go continuouslyPlay(matchIdGen, matchStats, matchChan)
	}

	// Rescore player1's moves, for learning.
	var rescoreMatchChan chan *Match
	if !isSamePlayer && *flagRescore {
		rescoreMatchChan = make(chan *Match, 5)
		for i := 0; i < parallelism; i++ {
			go continuouslyRescorePlayer1(matchChan, rescoreMatchChan)
		}
		// Swap matchChan and rescoreMatchChan, so that matchChan holds the
		// correctly labeled matches.
		matchChan, rescoreMatchChan = rescoreMatchChan, matchChan
	}

	// Batch matches.
	batchChan := make(chan []*Match, 2)
	go batchMatches(matchChan, batchChan)

	// Convert matches to labeled examples.
	labeledExamplesChan := make(chan LabeledBoards, 5)
	go continuousMatchesToLabeledExamples(batchChan, labeledExamplesChan)

	// Continuously learn.
	go continuousLearning(labeledExamplesChan)

	// Occasionally monitor queue sizes and save.
	ticker := time.NewTicker(300 * time.Second)
	lastSaveStep := p0GlobalStep()
	for _ = range ticker.C {
		if rescoreMatchChan == nil {
			klog.V(1).Infof("Queues: matches=%d batches=%d learning=%d",
				len(matchChan), len(batchChan), len(labeledExamplesChan))
		} else {
			klog.V(1).Infof("Queues: finishedMatches=%d rescoreMatches=%d batches=%d learning=%d",
				len(rescoreMatchChan), len(matchChan), len(batchChan), len(labeledExamplesChan))
		}
		globalStep := p0GlobalStep()
		if globalStep == -1 || globalStep > lastSaveStep {
			lastSaveStep = globalStep
			savePlayer0()
		}
	}
}

func continuouslyPlay(matchIdGen *IdGen, matchStats *MatchStats, matchChan chan<- *Match) {
	for {
		id := matchIdGen.NextId()
		match := runMatch(id)
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
		matchChan <- match
	}
}

func continuouslyRescorePlayer1(mInput <-chan *Match, mOutput chan<- *Match) {
	for match := range mInput {
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
		mOutput <- match
	}
}

func batchMatches(matchChan <-chan *Match, batchChan chan<- []*Match) {
	batchSize := *flagContinuosPlayAndTrainBatchMatches
	batch := make([]*Match, 0, batchSize)
	for match := range matchChan {
		batch = append(batch, match)
		klog.V(1).Infof("Current batch: %d", len(batch))
		if len(batch) == batchSize {
			klog.V(1).Infof("Batch of %d matches issued.", len(batch))
			batchChan <- batch
			batch = make([]*Match, 0, batchSize)
		}
	}
}

// continuousMatchesToLabeledExamples takes batches, and send labeled examples for training.
// It generate labeled examples from the latest 2 batches.
func continuousMatchesToLabeledExamples(batchChan <-chan []*Match, labeledExamplesChan chan<- LabeledBoards) {
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
