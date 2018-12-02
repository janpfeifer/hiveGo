package main

import (
	"github.com/janpfeifer/hiveGo/state"
	"log"
	"runtime"
	"sync"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
)

// Rescore the matches according to player 0 -- during rescoring we are only trying to
// improve the one AI.
func rescore(matches []*Match) {
	var wg sync.WaitGroup
	parallelism := runtime.GOMAXPROCS(0)
	if *flag_parallelism > 0 {
		parallelism = *flag_parallelism
	}
	setAutoBatchSizes(parallelism / 2)
	glog.V(1).Infof("Rescoring: parallelization=%d", parallelism)
	semaphore := make(chan bool, parallelism)
	for matchNum, match := range matches {
		wg.Add(1)
		semaphore <- true
		go func(matchNum int, match *Match) {
			defer wg.Done()
			defer func() { <-semaphore }()

			from := 0
			if *flag_lastActions > 0 && *flag_lastActions < len(match.Actions) {
				from = len(match.Actions) - *flag_lastActions
			}
			glog.V(2).Infof("lastActions=%d, from=%d, len(actions)=%d", *flag_lastActions, from, len(match.Actions))
			glog.V(2).Infof("Rescoring match %d", matchNum)
			newScores, actionsLabels := players[0].Searcher.ScoreMatch(
				match.Boards[from], match.Actions[from:len(match.Actions)],
				match.Boards[from:len(match.Boards)])
			copy(match.Scores[from:from+len(newScores)-1], newScores)
			copy(match.ActionsLabels[from:from+len(actionsLabels)], actionsLabels)
			for ii := from; ii < len(match.Actions); ii++ {
				if len(match.ActionsLabels[ii]) != match.Boards[ii].NumActions() {
					log.Panicf("Match %d: number of labels (%d) different than number of actions (%d) for move %d",
						match.MatchFileIdx, len(match.ActionsLabels[ii]), match.Boards[ii].NumActions(), ii)
				}
			}
			glog.V(2).Infof("Match %d (MatchFileIdx=%d) rescored.", matchNum, match.MatchFileIdx)
		}(matchNum, match)
	}

	// Gradually decrease the batching level.
	go func() {
		for ii := parallelism; ii > 0; ii-- {
			semaphore <- true
			setAutoBatchSizes(ii / 2)
		}
	}()

	// Wait for the remaining ones to finish.
	wg.Wait()
}

// trainFromExamples: only player[0] is trained.
func trainFromExamples(boards []*state.Board, boardLabels []float32, actionsLabels [][]float32) {
	learningRate := float32(*flag_learningRate)
	learn := func(steps int) float32 {
		return players[0].Learner.Learn(boards, boardLabels, actionsLabels, learningRate, steps)
	}
	log.Printf("Number of labeled examples: %d", len(boards))
	loss := learn(0)
	log.Printf("  Loss before train loop: %.2f", loss)
	if *flag_trainLoops > 0 {
		loss = learn(*flag_trainLoops)
		log.Printf("  Loss after %dth train loop: %.2f", *flag_trainLoops, loss)
		loss = learn(0)
		log.Printf("  Loss after train loop: %.2f", loss)
	}
	if players[0].ModelFile != "" {
		log.Printf("Saving to %s", players[0].ModelFile)
		ai.LinearModelFileName = players[0].ModelFile // Hack for linear models. TODO: fix.
		players[0].Learner.Save()
		if glog.V(1) {
			glog.V(1).Infof("%s", players[0].Learner)
		}
	}
}

func loopRescoreAndRetrainMatches(matchesChan chan *Match) {
	var matches []*Match
	for match := range matchesChan {
		matches = append(matches, match)
	}
	if len(matches) == 0 {
		log.Panic("No matches to rescore?!")
	}

	for rescoreIdx := 0; rescoreIdx < *flag_rescore; rescoreIdx++ {
		rescore(matches)
		var (
			boardExamples []*state.Board
			boardLabels   []float32
			actionsLabels [][]float32
		)
		for _, match := range matches {
			boardExamples, boardLabels, actionsLabels = match.AppendLabeledExamples(
				boardExamples, boardLabels, actionsLabels)
		}
		trainFromExamples(boardExamples, boardLabels, actionsLabels)
	}
}
