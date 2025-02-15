package main

import (
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"k8s.io/klog/v2"
	"math"
	"sync"
)

// Label matches with scores that reflect the final result.
// Also adds to the boards information of how many actions for
// the player till the end of the match, which is used to
// calculate the weighting based on TD-lambda constant.
func labelWithEndScore(match *Match) {
	_, endScore := ai.IsEndGameAndScore(match.FinalBoard())
	var scores [2]float32
	if match.FinalBoard().NextPlayer == 0 {
		scores = [2]float32{endScore, -endScore}
	} else {
		scores = [2]float32{-endScore, endScore}
	}
	numActions := len(match.Actions)
	for ii := range match.Scores {
		match.Boards[ii].Derived.PlayerMovesToEnd = int8((numActions-ii+1)/2 - 1)
		match.Scores[ii] = scores[match.Boards[ii].NextPlayer]
	}
}

// Rescore the matches according to player 0 -- during rescoring we are only trying to
// improve the one AI.
// Input and output are asynchronous: matches are started as they arrive, and are
// output as they are finished. If auto-batching is on, rescoring may be locked
// until enough matches are
func rescoreMatches(matchesIn <-chan *Match, matchesOut chan *Match) {
	var wg sync.WaitGroup
	parallelism := getParallelism()
	setAutoBatchSizes(parallelism / 2)
	klog.V(1).Infof("Rescoring: parallelization=%d", parallelism)
	semaphore := make(chan bool, parallelism)
	matchNum := 0
	for match := range matchesIn {
		wg.Add(1)
		semaphore <- true
		go func(matchNum int, match *Match) {
			defer wg.Done()
			defer func() { <-semaphore }()

			from, to, _ := match.SelectRangeOfActions()
			if from > len(match.Actions) {
				return
			}
			if to == -1 {
				// No selected action for this match.
				return
			}
			klog.V(2).Infof("lastActions=%d, from=%d, len(actions)=%d", *flagLastActions, from, len(match.Actions))
			klog.V(2).Infof("Rescoring match %d", matchNum)
			if *flagDistill {
				// Distillation: score boards.
				// TODO: For the scores just use the immediate score. Not action labels.
				newScores := directRescoreMatch(aiPlayers[1].Scorer, match, from, to)
				if len(newScores) > 0 {
					maxScore := math.Abs(float64(newScores[0]))
					for _, v := range newScores {
						if float64(v) > maxScore {
							maxScore = float64(v)
						}
					}
					klog.V(2).Infof("maxScore=%.2f", maxScore)
					copy(match.Scores[from:from+len(newScores)], newScores)
				}

			} else {
				// from -> to refer to the actions. ScoreMatch scores the boards up to the action following
				// the actions[to], hence one more than the number of actions.
				/* Old version with actions labels:
				newScores, ActionsLabels := aiPlayers[0].Searcher.ScoreMatch(
					match.Boards[from], match.Actions[from:to])
				copy(match.Scores[from:from+len(newScores)-1], newScores)
				copy(match.ActionsLabels[from:from+len(ActionsLabels)], ActionsLabels)
				*/
				for actionIdx := from; actionIdx <= to; actionIdx++ {
					_, _, score, _ := aiPlayers[0].Play(match.Boards[actionIdx])
					match.Scores[actionIdx] = score
				}

				// Sanity check of ActionsLabels.
				for ii := from; ii < len(match.Actions); ii++ {
					if match.ActionsLabels[ii] != nil && len(match.ActionsLabels[ii]) != match.Boards[ii].NumActions() {
						exceptions.Panicf("Match %d: number of labels (%d) different than number of actions (%d) for move %d",
							match.MatchFileIdx, len(match.ActionsLabels[ii]), match.Boards[ii].NumActions(), ii)
					}
				}
			}
			matchesOut <- match
			klog.V(2).Infof("Match %d (MatchFileIdx=%d) rescored.", matchNum, match.MatchFileIdx)
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
		leTrain      = &LabeledBoards{}
		leValidation = &LabeledBoards{}
	)
	for _, match := range matches {
		if *flagLearnWithEndScore {
			labelWithEndScore(match)
		}
		hashNum := match.FinalBoard().Derived.Hash
		if int(hashNum%100) >= *flagTrainValidation {
			match.AppendToLabeledBoards(leTrain)
		} else {
			match.AppendToLabeledBoards(leValidation)
		}
	}
	trainFromExamples(leTrain, leValidation)
}

func savePlayer0() error {
	if aiPlayers[0].Learner != nil {
		klog.Infof("Saving %s", aiPlayers[0].Learner)
		//features.LinearModelFileName = aiPlayers[0].Learner.ModelPath // Hack for linear models. TODO: fix.
		return aiPlayers[0].Learner.Save()
	}
	return nil
}

// trainFromExamples: only player[0] is trained.
func trainFromExamples(leTrain, leValidation *LabeledBoards) {
	//klog.Infof("Number of labeled examples: train=%d validation=%d", leTrain.Len(), leValidation.Len())
	//loss, boardLoss, actionsLoss := aiPlayers[0].Learner.Learn(
	//	leTrain.Boards, leTrain.Labels, leTrain.ActionsLabels,
	//	learningRate, epochs, perEpochCallback)
	for epoch := range *flagTrainLoops {
		trainLoss := aiPlayers[0].Learner.Loss(leTrain.Boards, leTrain.Labels)
		validLoss := aiPlayers[0].Learner.Loss(leValidation.Boards, leValidation.Labels)
		klog.Infof("  Epoch #%d losses: train=%.4g, validation=%.4g", epoch, trainLoss, validLoss)
	}
}

// directRescoreMatch scores the moves [from, to) of the given match using scorer.
// This can be used for distillation.
//
// It is "direct" because it doesn't use a searcher to do TD learning.
func directRescoreMatch(scorer ai.BoardScorer, match *Match, from, to int) (scores []float32) {
	boards := match.Boards[from:to]
	if len(boards) == 0 {
		return nil
	}
	batchScorer, ok := scorer.(ai.BatchBoardScorer)
	if ok {
		return batchScorer.BatchBoardScore(boards)
	}

	// Parallel score board positions.
	scores = make([]float32, len(boards))
	wg := &sync.WaitGroup{}
	for boardIdx, board := range boards {
		wg.Add(1)
		go func() {
			scores[boardIdx] = scorer.BoardScore(board)
			wg.Done()
		}()
	}
	wg.Wait()
	return
}
