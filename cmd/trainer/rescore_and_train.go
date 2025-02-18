package main

// This file implements continuous rescore and training of a database of matches.

import (
	"context"
	"fmt"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"math/rand"
	"time"
)

// MatchAction holds indices to Match/Action to be rescored.
type MatchAction struct {
	matchNum, actionNum int32
}

// RescoreAndTrain indefinitely pipes rescored board positions to a learner. Since the learner
// is updating the same model as used by the scorer, the model will constantly improve.
//
// Several flags control this process:
//
//   - -rescore_and_train: triggers this process.
//   - -rescore_pool_size: how many board positions to keep in pool. The larger the more times a rescored board
//     will be used for training.
func rescoreAndTrain(ctx context.Context, matches []*Match) error {
	parallelism := getParallelism()

	// Sample random match/action to rescore.
	maSampling := make(chan MatchAction, 2*parallelism)
	go sampleMatchActions(matches, maSampling)

	// Rescore the sampled actions.
	rescoredMA := make(chan MatchAction, 2*parallelism)
	for i := 0; i < parallelism; i++ {
		go rescoreMatchActions(matches, maSampling, rescoredMA)
	}

	// Collect rescored matches and issue learning.
	labeledExamplesChan := make(chan LabeledBoards, 2*parallelism)
	go collectMatchActionsAndIssueLearning(matches, rescoredMA, labeledExamplesChan)

	// Continuously learn.
	return continuousLearning(ctx, labeledExamplesChan)
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
			match.ActionsLabels[actionIdx] = actionsLabels // ActionsLabels[0]
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
	batchSize := aiPlayers[0].Learner.BatchSize()
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

const eraseToEndOfLine= "\033[K"

// learnSelection continuously learns from new matches.
//
// It yields the current learningSteps executed so far in learningSteps -- if learningSteps is not read, it may
// block the learning.
//
// If ctx is interrupted, learningSteps is closed and it exits.
func continuousLearning(ctx context.Context, matchesChan <-chan *Match) error {
	//var averageLoss, averageBoardLoss, averageActionsLoss float32
	var averageLoss float32
	labeledBoards := LabeledBoards{
		MaxSize: *flagTrainingBoardsBufferSize,
	}

	var countLearn, countMatches int
	lastSave := time.Now()
	var lastReport time.Time

	reportProgressFn := func() {
		fmt.Printf("\rProgress: #matches=%d, #steps=%d, ~loss=%.4g%s",
			countMatches, countLearn, averageLoss, eraseToEndOfLine)
	}
	// Makes sure we update the progress before exiting (even in case of error).
	defer func() {
		reportProgressFn()
		fmt.Println()
	}()

	for {
		klog.V(3).Infof("Learn: #matches=%d, #learn=%d, ~loss=%.4g",
			countMatches, countLearn, averageLoss)

		// Read next match or exit if no more matches or if ctx has been interrupted.
		// TODO: create an iterator for this in generics.
		var match *Match
		var ok bool
		select {
		case <-ctx.Done():
			return errors.WithMessagef(ctx.Err(), "execution interrupted")
		case match, ok = <-matchesChan:
			if !ok {
				// End of matches.
				return nil
			}
		}
		countMatches++

		if labeledBoards.Len() < labeledBoards.MaxSize {
			// Only filling up buffer before we start training.
			continue
		}

		// First learn with 0 steps: only evaluation without dropout.
		//loss, boardLoss, actionsLoss := aiPlayers[0].Learner.Learn(
		//	le.Boards, le.Labels,
		//	le.ActionsLabels, float32(*flagLearningRate),
		//	0, nil)
		//klog.V(1).Infof("Evaluation loss: total=%g board=%g actions=%g",
		//	loss, boardLoss, actionsLoss)
		loss := aiPlayers[0].Learner.Loss(labeledBoards.Boards, labeledBoards.Labels)
		klog.V(1).Infof("Pre-training loss %.4g", loss)

		// Actually train.
		//_, _, _ = aiPlayers[0].Learner.Learn(
		//	le.Boards, le.Labels,
		//	le.ActionsLabels, float32(*flagLearningRate),
		//	*flagTrainLoops, nil)
		loss = aiPlayers[0].Learner.Loss(labeledBoards.Boards, labeledBoards.Labels)
		countLearn++
		klog.V(1).Infof("Post-training loss %.4g", loss)

		// Evaluate (learn with 0 steps)on training data.
		//loss, boardLoss, actionsLoss = aiPlayers[0].Learner.Learn(
		//	le.Boards, le.Labels,
		//	le.ActionsLabels, float32(*flagLearningRate),
		//	0, nil)
		//klog.V(1).Infof("Training loss: total=%g board=%g actions=%g",
		//	loss, boardLoss, actionsLoss)

		decay := 1 - 1/float32(1+countLearn)
		if decay > maxAverageLossDecay {
			decay = maxAverageLossDecay
		}
		averageLoss = decayAverageLoss(averageLoss, loss, decay)
		//averageBoardLoss = decayAverageLoss(averageBoardLoss, boardLoss, decay)
		//averageActionsLoss = decayAverageLoss(averageActionsLoss, actionsLoss, decay)

		// Report progress:
		if time.Since(lastReport).Seconds() > 1 {
			lastReport = time.Now()
		}

			if time.Since(lastSave).Seconds() > 60 {
				if err := savePlayer0(); err != nil {
					return err
				}
				klog.V(1).Infof("Saved model at step=%d", stepNum)
				lastSave = time.Now()
			}
		}
	}
}

const maxAverageLossDecay = float32(0.95)

func decayAverageLoss(average, newValue, decay float32) float32 {
	return average*decay + (1-decay)*newValue
}
