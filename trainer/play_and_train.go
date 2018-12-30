package main

// This file implements continuous play and train.

import (
	"sync"
	"time"

	"github.com/golang/glog"
	. "github.com/janpfeifer/hiveGo/state"
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

	// Convert matches to labeled examples.
	labeledExamplesChan := make(chan LabeledExamples, 2*parallelism)
	go continuousMatchesToLabeledExamples(matchChan, labeledExamplesChan)

	// Continuously learn.
	go continuousLearning(labeledExamplesChan)

	// Occasionally monitor queue sizes and save.
	ticker := time.NewTicker(300 * time.Second)
	lastSaveStep := p0GlobalStep()
	for _ = range ticker.C {
		glog.V(1).Infof("Queues: matches=%d, learning=%d", len(matchChan), len(labeledExamplesChan))
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
		glog.V(1).Infof("Match %d finished. %d matches played so far. Last %d results: p0 win=%d, p1 win=%d, draw=%d",
			id, matchStats.TotalCount, NumMatchesToKeepForStats,
			matchStats.Wins[0], matchStats.Wins[1], matchStats.Draws)
		matchChan <- match
	}
}

func continuousMatchesToLabeledExamples(matchChan <-chan *Match, labeledExamplesChan chan<- LabeledExamples) {
	for match := range matchChan {
		// Prepare set of labeled examples.
		n := len(match.Actions)
		if !isSamePlayer {
			n = (n + 1) / 2
		}
		le := LabeledExamples{
			boardExamples: make([]*Board, 0, n),
			boardLabels:   make([]float32, 0, n),
			actionsLabels: make([][]float32, 0, n),
		}

		// Pick only moves done by player 0 (or both if they are the same)
		for idx := range match.Actions {
			board := match.Boards[idx]
			player := board.NextPlayer
			if match.Swapped {
				player = 1 - player
			}
			if player == 0 || isSamePlayer {
				if board.NumActions() < 2 {
					// Skip when there is none or only one action available.
					continue
				}
				le.boardExamples = append(le.boardExamples, board)
				le.boardLabels = append(le.boardLabels, match.Scores[idx])
				le.actionsLabels = append(le.actionsLabels, match.ActionsLabels[idx])
			}
		}
		labeledExamplesChan <- le
	}
}
