package main

// This file implements continuous play and train.

import (
	"flag"
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"
	. "github.com/janpfeifer/hiveGo/state"
)

var (
	flag_continuosPlayAndTrain = flag.Bool("play_and_train",
		false, "If set, continuously play and train matches.")
	flag_continuosPlayAndTrainBatchMatches = flag.Int("play_and_train_batch_matches",
		10, "Number of matches to batch before learning.")
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

const NumMatchesToKeepForStats = 100

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
		glog.Infof("Same player playing both sides, learning from both.")
	}

	// Generate played games.
	matchChan := make(chan *Match, 2*parallelism)
	matchIdGen := &IdGen{}
	matchStats := NewMatchStats()
	for i := 0; i < parallelism; i++ {
		go continuouslyPlay(matchIdGen, matchStats, matchChan)
	}

	// Rescore player1's moves, for learning.
	rescoreMatchChan := make(chan *Match, 2*parallelism)
	if !isSamePlayer {
		for i := 0; i < parallelism; i++ {
			go continuouslyRescorePlayer1(matchChan, rescoreMatchChan)
		}
		// Swap matchChan and rescoreMatchChan, so that matchChan holds the
		// correctly labeled matches.
		matchChan, rescoreMatchChan = rescoreMatchChan, matchChan
	}

	// Batch matches.
	batchChan := make(chan []*Match, 2*parallelism)
	go batchMatches(matchChan, batchChan)

	// Convert matches to labeled examples.
	labeledExamplesChan := make(chan LabeledExamples, 2*parallelism)
	for i := 0; i < parallelism; i++ {
		go continuousMatchesToLabeledExamples(batchChan, labeledExamplesChan)
	}

	// Continuously learn.
	go continuousLearning(labeledExamplesChan)

	// Occasionally monitor queue sizes and save.
	ticker := time.NewTicker(300 * time.Second)
	lastSaveStep := p0GlobalStep()
	for _ = range ticker.C {
		if isSamePlayer {
			glog.V(1).Infof("Queues: matches=%d learning=%d", len(matchChan), len(labeledExamplesChan))
		} else {
			glog.V(1).Infof("Queues: finishedMatches=%d rescoreMatches=%d learning=%d", len(rescoreMatchChan), len(matchChan), len(labeledExamplesChan))
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
		glog.V(1).Infof("Match %d finished (%s in %d moves). %d matches played so far. "+
			"Last %d results: p0 win=%d, p1 win=%d, draw=%d",
			id, msg, len(match.Actions), matchStats.TotalCount, NumMatchesToKeepForStats,
			matchStats.Wins[0], matchStats.Wins[1], matchStats.Draws)
		matchChan <- match
	}
}

func continuouslyRescorePlayer1(mInput <-chan *Match, mOutput chan<- *Match) {
	for match := range mInput {
		// Pick only moves done by player 0 (or both if they are the same)
		for idx := range match.Actions {
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

			// Rescore action.
			newScores, actionsLabels := players[0].Searcher.ScoreMatch(board, match.Actions[idx:idx+1])

			// Copy over new scores and labels for the particular action on the match.
			match.Scores[idx] = newScores[0]
			if actionsLabels != nil {
				match.ActionsLabels[idx] = actionsLabels[0]
			}
		}
		mOutput <- match
	}
}

func batchMatches(matchChan <-chan *Match, batchChan chan<- []*Match) {
	batchSize := *flag_continuosPlayAndTrainBatchMatches
	batch := make([]*Match, 0, batchSize)
	for match := range matchChan {
		batch = append(batch, match)
		if len(batch) == batchSize {
			batchChan <- batch
			batch = make([]*Match, 0, batchSize)
		}
	}
}

func continuousMatchesToLabeledExamples(batchChan <-chan []*Match, labeledExamplesChan chan<- LabeledExamples) {
	for batch := range batchChan {
		n := 0
		for _, match := range batch {
			n += len(match.Actions)
		}
		le := LabeledExamples{
			boardExamples: make([]*Board, 0, n),
			boardLabels:   make([]float32, 0, n),
			actionsLabels: make([][]float32, 0, n),
		}
		for _, match := range batch {
			match.AppendLabeledExamples(&le)
		}
		labeledExamplesChan <- le
	}
}

