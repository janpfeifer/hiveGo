package main

import (
	"math/rand"
	"runtime"
	"time"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai/tensorflow"
	. "github.com/janpfeifer/hiveGo/state"
)

// MatchAction holds indices to Match/Action to be rescored.
type MatchAction struct {
	matchNum, actionNum int32
}

// LearnParams holds parameters of one batch to learn.
type LearnParams struct {
	boards        []*Board
	boardLabels   []float32
	actionsLabels [][]float32
}

// RescoreAndTrain indefinitely pipes rescored board positions to a learner. Since the learner
// is updating the same model as used by the scorer, the model will constantly improve.
//
// Several flags control this process:
// --rescore_and_train: triggers this process.
// --rescore_pool_size: how many board positions to keep in pool. The larger the more times a
//   rescored board will be used for training. It must be larger than --tf_batch_size.
func rescoreAndTrain(matches []*Match) {
	parallelism := runtime.GOMAXPROCS(0)
	if *flag_parallelism > 0 {
		parallelism = *flag_parallelism
	}

	// Sample random match/action to rescore.
	maSampling := make(chan MatchAction, 2*parallelism)
	go sampleMatchActions(matches, maSampling)

	// Rescore the sampled actions.
	rescoredMA := make(chan MatchAction, 2*parallelism)
	for i := 0; i < parallelism; i++ {
		go rescoreMatchActions(matches, maSampling, rescoredMA)
	}

	// Collect rescored matches and issue learning.
	learnParamsChan := make(chan LearnParams, 2*parallelism)
	go collectMatchActionsAndIssueLearning(matches, rescoredMA, learnParamsChan)

	// Continuously learn.
	go continuousLearning(learnParamsChan)

	ticker := time.NewTicker(60 * time.Second)
	for _ = range ticker.C {
		glog.V(1).Infof("Queues: sampling=%d, rescoring=%d, learning=%d",
			len(maSampling), len(rescoredMA), len(learnParamsChan))
		savePlayer0()
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
	for ma := range maInput {
		match := matches[ma.matchNum]
		from := int(ma.actionNum)
		to := from + 1
		newScores, actionsLabels := players[0].Searcher.ScoreMatch(
			match.Boards[from], match.Actions[from:to])

		// Copy over new scores and labels for the particular action on the match.
		match.mu.Lock()
		if match.Scores == nil {
			match.Scores = make([]float32, len(match.Boards))
		}
		if match.ActionsLabels == nil {
			match.ActionsLabels = make([][]float32, len(match.Actions))
		}

		match.Scores[from] = newScores[0]
		if actionsLabels != nil {
			match.ActionsLabels[from] = actionsLabels[0]
		}
		match.mu.Unlock()

		maOutput <- ma
	}
}

func MakeLearnParams(batchSize int) LearnParams {
	return LearnParams{
		boards:        make([]*Board, 0, batchSize),
		boardLabels:   make([]float32, 0, batchSize),
		actionsLabels: make([][]float32, 0, batchSize),
	}
}

func (l *LearnParams) Append(matches []*Match, ma MatchAction) {
	match := matches[ma.matchNum]
	l.boards = append(l.boards, match.Boards[ma.actionNum])
	l.boardLabels = append(l.boardLabels, match.Scores[ma.actionNum])
	l.actionsLabels = append(l.actionsLabels, match.ActionsLabels[ma.actionNum])
}

func collectMatchActionsAndIssueLearning(matches []*Match, maInput <-chan MatchAction, learnOutput chan<- LearnParams) {
	poolSize := *flag_rescoreAndTrainPoolSize
	batchSize := *tensorflow.Flag_learnBatchSize
	issueFreq := *flag_rescoreAndTrainIssueLearn

	pool := make([]MatchAction, 0)
	count := 0
	for ma := range maInput {
		// Append new MatchAction or, if pool is full, start rotating them.
		if len(pool) < poolSize {
			pool = append(pool, ma)
		} else {
			pool[count%poolSize] = ma
		}
		count++
		glog.V(3).Infof("Pool=%d, count=%d, batchSize=%d", len(pool), count, batchSize)

		if count >= batchSize {
			if count <= 10*batchSize {
				// Until enough examples are collected, only learn with newly rescored
				// results.
				if count%batchSize == 0 {
					lp := MakeLearnParams(batchSize)
					for ii := 0; ii < batchSize; ii++ {
						idx := (count - 1 - ii) % poolSize
						lp.Append(matches, pool[idx])
					}
					learnOutput <- lp
				}
			} else if count%issueFreq == 0 {
				// After we have enough examples, train at every few new
				// rescored examples, plus some random ones.
				lp := MakeLearnParams(batchSize)
				for ii := 0; ii < issueFreq; ii++ {
					idx := (count - 1 - ii) % poolSize
					lp.Append(matches, pool[idx])
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
					lp.Append(matches, pool[idx])
				}
				learnOutput <- lp
			}
		}

	}
}

// learnSelection continuously learns from the selected board data.
func continuousLearning(learnInput <-chan LearnParams) {
	var averageLoss, averageBoardLoss, averageActionsLoss float32
	count := 0
	for params := range learnInput {
		glog.V(3).Infof("Learn: count=%d", count)
		loss, boardLoss, actionsLoss := players[0].Learner.Learn(
			params.boards, params.boardLabels,
			params.actionsLabels, float32(*flag_learningRate),
			1, nil)
		glog.V(2).Infof("Losses: total=%g board=%g actions=%g", loss, boardLoss, actionsLoss)
		decay := 1 / float32(1+count)
		if decay > averageLossDecay {
			decay = averageLossDecay
		}
		averageLoss = decayAverageLoss(averageLoss, loss, decay)
		averageBoardLoss = decayAverageLoss(averageBoardLoss, boardLoss, decay)
		averageActionsLoss = decayAverageLoss(averageActionsLoss, actionsLoss, decay)
		if glog.V(2) || count%100 == 0 {
			globalStep := int64(-1)
			if tf := players[0].Learner.(*tensorflow.Scorer); tf != nil {
				globalStep = tf.ReadGlobalStep()
			}
			glog.Infof("Average Losses (step=%d): total=%.4g board=%.4g actions=%.4g",
				globalStep, averageLoss, averageBoardLoss, averageActionsLoss)
		}
		count++
	}
}

const averageLossDecay = float32(0.99)

func decayAverageLoss(average, newValue, decay float32) float32 {
	return average*decay + (1-decay)*newValue
}
