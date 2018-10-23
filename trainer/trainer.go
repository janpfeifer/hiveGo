package main

import (
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
	glog.V(1).Infof("Rescoring: parallelization=%d", runtime.GOMAXPROCS(0))
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
			glog.V(2).Infof("Rescoring match %d", matchNum)
			newScores := players[0].Searcher.ScoreMatch(match.Boards[from], match.Actions[from:len(match.Actions)])
			copy(match.Scores[from:from+len(newScores)-1], newScores)
			glog.V(2).Infof("Match %d rescored.", matchNum)
		}(matchNum, match)
	}

	// Wait for the remaining ones to finish.
	wg.Wait()
}

// trainFromExamples: only player[0] is trained.
func trainFromExamples(labeledExamples []ai.LabeledExample) {
	learningRate := float32(*flag_learningRate)
	log.Printf("Number of labeled examples: %d", len(labeledExamples))
	loss := players[0].Learner.Learn(learningRate, labeledExamples, 0)
	log.Printf("  Loss before train loop: %.2f", loss)
	if *flag_trainLoops > 0 {
		loss = players[0].Learner.Learn(learningRate, labeledExamples, *flag_trainLoops)
		log.Printf("  Loss after %dth train loop: %.2f", *flag_trainLoops, loss)
		loss = players[0].Learner.Learn(learningRate, labeledExamples, 0)
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

	for rescoreIdx := 0; rescoreIdx < *flag_rescore; rescoreIdx++ {
		rescore(matches)
		var labeledExamples []ai.LabeledExample
		for _, match := range matches {
			labeledExamples = match.AppendLabeledExamples(labeledExamples)
		}
		trainFromExamples(labeledExamples)
	}
}
