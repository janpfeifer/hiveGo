package main

// This file implements continuous rescore and training of a database of matches.

import (
	"context"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"math/rand"
)

// MatchAction holds indices to Match/Action to be rescored.
type MatchAction struct {
	matchNum, actionNum int32
}

// RescoreAndTrain indefinitely pipes rescored board positions to a learner. Since the learner
// is updating the same model as used by the scorer, the model will constantly improve.
//
// Flags that control this process:
//
//   - -rescore_and_train: triggers this process.
//   - -train_buffer_size: specifies the size of the rotating buffer from which to draw batches to train.
func rescoreAndTrain(ctx context.Context, matches []*Match) error {
	parallelism := getParallelism()

	// Sample random matches.
	matchesChan := make(chan *Match, 2*parallelism)
	go sampleMatches(ctx, matches, matchesChan)

	// Rescore matches.
	rescoreMatchesChan := make(chan *Match, 2*parallelism)
	for i := 0; i < parallelism; i++ {
		go continuouslyRescoreMatches(ctx, matchesChan, rescoreMatchesChan)
	}

	// Continuously learn.
	return continuousLearning(ctx, matchesChan)
}

func sampleMatches(ctx context.Context, matches []*Match, matchesChan chan<- *Match) {
	for {
		matchNum := rand.Int31n(int32(len(matches)))
		select {
		case <-ctx.Done():
			klog.Infof("sampleMatches() interrupted: context cancelled with %v", ctx.Err())
			return
		case matchesChan <- matches[matchNum]:
			// Move to next.
		}
	}
}

// sampleMatchActions continuously sample random matches/actions
func sampleMatchActions(matches []*Match, maSampling chan<- MatchAction) {
	for {
		ma := MatchAction{matchNum: rand.Int31n(int32(len(matches)))}
		match := matches[ma.matchNum]
		ma.actionNum = rand.Int31n(int32(len(match.Actions)))
		board := match.Boards[ma.actionNum]
		if board.NumActions() < 2 {
			// We don't sample when there are only one or no action
			// available.
			continue
		}
		maSampling <- ma
	}
}

// rescoreMatchActions rescores each match/action, and outputs it.
func rescoreMatchActions(matches []*Match, maInput <-chan MatchAction, maOutput chan<- MatchAction) {
	for matchAction := range maInput {
		match := matches[matchAction.matchNum]
		actionIdx := int(matchAction.actionNum)
		//to := from + 1
		//newScores, ActionsLabels := aiPlayers[0].Searcher.ScoreMatch(
		//	match.Boards[from], match.Actions[from:to])
		_, _, newScore, actionsLabels := aiPlayers[0].Play(match.Boards[actionIdx])
		match.Boards[actionIdx].ClearNextBoardsCache() // Clear sub-tree generated by match.

		// Clone over new scores and labels for the particular action on the match.
		match.mu.Lock()
		if match.Scores == nil {
			match.Scores = make([]float32, len(match.Boards))
		}
		match.Scores[actionIdx] = newScore
		if actionsLabels != nil {
			if match.ActionsLabels == nil {
				match.ActionsLabels = make([][]float32, len(match.Actions))
			}
			match.ActionsLabels[actionIdx] = actionsLabels
		}
		match.mu.Unlock()
		maOutput <- matchAction
	}
}

func MakeLabeledExamples(batchSize int) LabeledBoards {
	return LabeledBoards{
		Boards:        make([]*Board, 0, batchSize),
		Labels:        make([]float32, 0, batchSize),
		ActionsLabels: make([][]float32, 0, batchSize),
	}
}

func (lb *LabeledBoards) AppendMatchAction(matches []*Match, ma MatchAction) {
	match := matches[ma.matchNum]
	lb.Boards = append(lb.Boards, match.Boards[ma.actionNum])
	lb.Labels = append(lb.Labels, match.Scores[ma.actionNum])
	lb.ActionsLabels = append(lb.ActionsLabels, match.ActionsLabels[ma.actionNum])
}

func collectMatchActionsAndIssueLearning(matches []*Match, maInput <-chan MatchAction, learnOutput chan<- LabeledBoards) {
	poolSize := *flagRescoreAndTrainPoolSize
	//batchSize := *tensorflow.Flag_learnBatchSize
	batchSize := aiPlayers[0].ValueLearner.BatchSize()
	issueFreq := *flagRescoreAndTrainIssueLearn

	pool := make([]MatchAction, 0)
	count := 0
	for ma := range maInput {
		// AddBoard new MatchAction or, if pool is full, start rotating them.
		if len(pool) < poolSize {
			pool = append(pool, ma)
		} else {
			pool[count%poolSize] = ma
		}
		count++
		klog.V(3).Infof("Pool=%d, count=%d, batchSize=%d", len(pool), count, batchSize)

		if count >= batchSize {
			if count <= 10*batchSize {
				// Until enough examples are collected, only learn with newly rescored
				// results.
				if count%batchSize == 0 {
					lp := MakeLabeledExamples(batchSize)
					for ii := 0; ii < batchSize; ii++ {
						idx := (count - 1 - ii) % poolSize
						lp.AppendMatchAction(matches, pool[idx])
					}
					learnOutput <- lp
				}
			} else if count%issueFreq == 0 {
				// After we have enough examples, train at every few new
				// rescored examples, plus some random ones.
				lp := MakeLabeledExamples(batchSize)
				for ii := 0; ii < issueFreq; ii++ {
					idx := (count - 1 - ii) % poolSize
					lp.AppendMatchAction(matches, pool[idx])
				}
				for ii := 0; ii < batchSize-issueFreq; ii++ {
					r := rand.Float64()
					r = r * r
					if count > poolSize {
						r *= float64(poolSize)
					} else {
						r *= float64(count)
					}
					idx := (count - 1 - int(r)) % poolSize
					lp.AppendMatchAction(matches, pool[idx])
				}
				learnOutput <- lp
			}
		}

	}
}
