package main

import (
	"runtime"
	"sync"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
)

// Rescore the matches according to player 0 -- during rescoring we are only trying to
// improve the one AI.
func rescore(matches []*Match, reuse bool) {
	var wg sync.WaitGroup
	semaphore := make(chan bool, runtime.GOMAXPROCS(0))
	for matchNum, match := range matches {
		wg.Add(1)
		semaphore <- true
		go func(matchNum int, match *Match) {
			defer wg.Done()
			defer func() { <-semaphore }()

			from := 0
			if *flag_lastActions > 1 && *flag_lastActions < len(match.Actions) {
				from = len(match.Actions) - *flag_lastActions
			}
			newScores := players[0].Searcher.ScoreMatch(
				match.Boards[from], players[0].Scorer,
				match.Actions[from:len(match.Actions)], reuse)
			copy(match.Scores[from:from+len(newScores)-1], newScores)
			glog.V(1).Infof("Match %d rescored.", matchNum)
		}(matchNum, match)
	}

	// Wait for the remaining ones to finish.
	wg.Wait()
}

// trainFromExamples: only player[0] is trained.
func trainFromExamples(labeledExamples []ai.LabeledExample) {
	const learningRate = 1e-5
	glog.V(1).Infof("len(LabeledExamples)=%d", len(labeledExamples))
	loss := players[0].Learner.Learn(learningRate, labeledExamples, 0)
	glog.Infof("  Loss before train loop: %.2f", loss)
	if *flag_trainLoops > 0 {
		loss = players[0].Learner.Learn(learningRate, labeledExamples, *flag_trainLoops)
		glog.Infof("  Loss after %dth train loop: %.2f", *flag_trainLoops, loss)
	}
	if players[0].ModelFile != "" {
		glog.Infof("Saving to %s", players[0].ModelFile)
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

	for rescoreIdx := 0; rescoreIdx < *flag_rescore; rescoreIdx++ {
		rescore(matches, *flag_rescore > 1)
		var labeledExamples []ai.LabeledExample
		for _, match := range matches {
			labeledExamples = match.AppendLabeledExamples(labeledExamples)
		}
		trainFromExamples(labeledExamples)
	}
}
