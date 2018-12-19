package main

import (
	"log"
	"math"
	"runtime"
	"sync"

	"github.com/janpfeifer/hiveGo/state"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
)

// Rescore the matches according to player 0 -- during rescoring we are only trying to
// improve the one AI.
// Input and output are asynchronous: matches are started as they arrive, and are
// output as they are finished. If auto-batching is on, rescoring may be locked
// until enough matches are
func rescoreMatches(matchesIn <-chan *Match, matchesOut chan *Match) {
	var wg sync.WaitGroup
	parallelism := runtime.GOMAXPROCS(0)
	if *flag_parallelism > 0 {
		parallelism = *flag_parallelism
	}
	setAutoBatchSizes(parallelism / 2)
	glog.V(1).Infof("Rescoring: parallelization=%d", parallelism)
	semaphore := make(chan bool, parallelism)
	matchNum := 0
	for match := range matchesIn {
		wg.Add(1)
		semaphore <- true
		go func(matchNum int, match *Match) {
			defer wg.Done()
			defer func() { <-semaphore }()

			from, to := match.SelectRangeOfActions()
			if from > len(match.Actions) {
				return
			}
			if to == -1 {
				// No selected action for this match.
				return
			}
			glog.V(2).Infof("lastActions=%d, from=%d, len(actions)=%d", *flag_lastActions, from, len(match.Actions))
			glog.V(2).Infof("Rescoring match %d", matchNum)
			if *flag_distill {
				// Distillation: score boards.
				// TODO: For the scores just use the immediate score. Not action labels.
				newScores := distill(players[1].Scorer, match, from, to)
				if len(newScores) > 0 {
					maxScore := math.Abs(float64(newScores[0]))
					for _, v := range newScores {
						if float64(v) > maxScore {
							maxScore = float64(v)
						}
					}
					glog.V(2).Infof("maxScore=%.2f", maxScore)
					copy(match.Scores[from:from+len(newScores)], newScores)
				}
			} else {
				// from -> to refer to the actions. ScoreMatch scores the boards up to the action following
				// the actions[to], hence one more than the number of actions.
				newScores, actionsLabels := players[0].Searcher.ScoreMatch(
					match.Boards[from], match.Actions[from:to], match.Boards[from:to+1])
				copy(match.Scores[from:from+len(newScores)-1], newScores)
				copy(match.ActionsLabels[from:from+len(actionsLabels)], actionsLabels)
				for ii := from; ii < len(match.Actions); ii++ {
					if match.ActionsLabels[ii] != nil && len(match.ActionsLabels[ii]) != match.Boards[ii].NumActions() {
						log.Panicf("Match %d: number of labels (%d) different than number of actions (%d) for move %d",
							match.MatchFileIdx, len(match.ActionsLabels[ii]), match.Boards[ii].NumActions(), ii)
					}
				}
			}
			matchesOut <- match
			glog.V(2).Infof("Match %d (MatchFileIdx=%d) rescored.", matchNum, match.MatchFileIdx)
		}(matchNum, match)
		matchNum++
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
	close(matchesOut)
}

func trainFromMatches(matches []*Match) {
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

// trainFromExamples: only player[0] is trained.
func trainFromExamples(boards []*state.Board, boardLabels []float32, actionsLabels [][]float32) {
	learningRate := float32(*flag_learningRate)
	learn := func(steps int) float32 {
		return players[0].Learner.Learn(boards, boardLabels, actionsLabels, learningRate, steps)
	}
	log.Printf("Number of labeled examples: %d", len(boards))
	loss := learn(0)
	log.Printf("  Loss before train loop: %.4g", loss)
	if *flag_trainLoops > 0 {
		loss = learn(*flag_trainLoops)
		log.Printf("  Loss after %dth train loop: %.4g", *flag_trainLoops, loss)
		loss = learn(0)
		log.Printf("  Loss after train loop: %.4g", loss)
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

// distill returns the score of the given board position, and no
// action labels.
func distill(scorer ai.BatchScorer, match *Match, from, to int) (scores []float32) {
	boards := match.Boards[from:to]
	if len(boards) == 0 {
		return nil
	}
	scores, _ = scorer.BatchScore(boards, false)
	return
}
