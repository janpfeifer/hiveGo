package main

import (
	"context"
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"math"
	"math/rand"
	"sync"
	"time"
)

// This file implements the training once one has the training data (LabeledBoards).
//
// It also defines LabeledBoards, the container of with the boards and labels to train.

// LabeledBoards is a container of board positions and its labels (scores), used for training.
//
// ActionsLabels are optional. If set, they must match the number of actions of the corresponding board.
type LabeledBoards struct {
	Boards        []*Board
	Labels        []float32
	ActionsLabels [][]float32

	// MaxSize configures the max number of boards to hold in LabeledBoards, if > 0.
	// After it reaches the MaxSize new boards appended start rotating the position (replacing older ones).
	MaxSize, CurrentIdx int
}

// Len returns the number of board positions stored.
func (lb *LabeledBoards) Len() int {
	return len(lb.Boards)
}

// AddBoard and its labels to LabeledBoards collection.
// If LabeledBoards has a MaxSize configured, and it is full, it starts recycling its buffer of boards.
func (lb *LabeledBoards) AddBoard(board *Board, label float32, actionsLabels []float32) {
	if lb.MaxSize == 0 || lb.Len() < lb.MaxSize {
		// AddBoard to the end.
		lb.Boards = append(lb.Boards, board)
		lb.Labels = append(lb.Labels, label)
		if actionsLabels != nil {
			lb.ActionsLabels = append(lb.ActionsLabels, actionsLabels)
		}
	} else {
		// Start cycling current buffer.
		lb.CurrentIdx = lb.CurrentIdx % lb.MaxSize
		lb.Boards[lb.CurrentIdx] = board
		lb.Labels[lb.CurrentIdx] = label
		if actionsLabels != nil {
			lb.ActionsLabels[lb.CurrentIdx] = actionsLabels
		}
	}
	lb.CurrentIdx++
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

const eraseToEndOfLine = "\033[K"

// learnSelection continuously learns from new matches.
//
// It yields the current learningSteps executed so far in learningSteps -- if learningSteps is not read, it may
// block the learning.
//
// If ctx is interrupted, learningSteps is closed and it exits.
//
// It uses the flag -train_buffer_size to specify the size of the rotating buffer to store matches.
func continuousLearning(ctx context.Context, matchesChan <-chan *Match) error {
	//var averageLoss, averageBoardLoss, averageActionsLoss float32
	var averageLoss float32
	labeledBoards := LabeledBoards{
		MaxSize: *flagTrainBoardsBufferSize,
	}
	batchSize := aiPlayers[0].Learner.BatchSize()
	numLearnSteps := *flagTrainStepsPerMatch
	if numLearnSteps == 0 {
		numLearnSteps = (*flagTrainBoardsBufferSize + batchSize - 1) / batchSize
	}

	var countLearn, countMatches int
	var ratioWins, ratioLosses, ratioDraws float32
	lastSave := time.Now()
	var lastReport time.Time

	reportProgressFn := func() {
		fmt.Printf("\rProgress: #matches=%d (~wins=%.1f%%, ~losses=%.1f%%, ~draws=%.1f%%), #steps=%d, ~loss=%.4g%s",
			countMatches,
			100*ratioWins,
			100*ratioLosses,
			100*ratioDraws,
			countLearn, averageLoss, eraseToEndOfLine)
	}
	// Makes sure we update the progress before exiting (even in case of error).
	defer func() {
		reportProgressFn()
		fmt.Println()
	}()

	for match := range generics.IterChanWithContext(ctx, matchesChan) {
		countMatches++
		klog.V(3).Infof("Learn: #matches=%d, #learn=%d, ~loss=%.4g",
			countMatches, countLearn, averageLoss)
		updateMovingAverages(match, countMatches, &ratioWins, &ratioLosses, &ratioDraws)

		match.AppendToLabeledBoards(&labeledBoards)
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
		if klog.V(1).Enabled() {
			preTrainLoss := aiPlayers[0].Learner.Loss(labeledBoards.Boards, labeledBoards.Labels)
			klog.Infof("Pre-training loss %.4g", preTrainLoss)
		}

		boardsBatch := make([]*Board, batchSize)
		labelsBatch := make([]float32, batchSize)
		var loss float32
		for range numLearnSteps {
			// Sample batch: random with replacement:
			for exampleIdx := range batchSize {
				idx := rand.Intn(labeledBoards.Len())
				boardsBatch[exampleIdx] = labeledBoards.Boards[idx]
				labelsBatch[exampleIdx] = labeledBoards.Labels[idx]
			}
			loss = aiPlayers[0].Learner.Learn(boardsBatch, labelsBatch)
			countLearn++
		}
		klog.V(1).Infof("Post-training loss %.4g", loss)

		// Evaluate (learn with 0 steps)on training data.
		//loss, boardLoss, actionsLoss = aiPlayers[0].Learner.Learn(
		//	le.Boards, le.Labels,
		//	le.ActionsLabels, float32(*flagLearningRate),
		//	0, nil)
		//klog.V(1).Infof("Training loss: total=%g board=%g actions=%g",
		//	loss, boardLoss, actionsLoss)

		averageLoss = movingAverage(averageLoss, loss, averageLossDecay, countLearn)
		//averageBoardLoss = movingAverage(averageBoardLoss, boardLoss, decay)
		//averageActionsLoss = movingAverage(averageActionsLoss, actionsLoss, decay)

		// Report progress:
		if time.Since(lastReport).Seconds() > 1 {
			reportProgressFn()
			lastReport = time.Now()
		}

		if time.Since(lastSave).Seconds() > 60 {
			fmt.Println()
			if err := savePlayer0(); err != nil {
				return err
			}
			klog.V(1).Infof("Saved model: #matches=%d, #steps=%d, ~loss=%.4g%s",
				countMatches, countLearn, averageLoss, eraseToEndOfLine)
			lastSave = time.Now()
		}
	}
	return nil
}

const averageLossDecay = float32(0.95)

func movingAverage(average, newValue, decay float32, count int) float32 {
	decay = min(1-1/float32(count), decay)
	return average*decay + (1-decay)*newValue
}

const ratiosDecay = 0.99 // for moving averages.

func updateMovingAverages(match *Match, count int, ratioWins, ratioLosses, ratioDraws *float32) {
	winner := match.FinalBoard().Winner()
	var valueDraw, valueWin, valueLose float32
	if winner == PlayerInvalid {
		valueDraw = 1
	} else {
		if match.Swapped {
			winner = 1 - winner
		}
		switch winner {
		case PlayerFirst:
			valueWin = 1
		case PlayerSecond:
			valueLose = 1
		default:
			// Draw already accounted for.
		}
	}
	// Update moving averages
	*ratioDraws = movingAverage(*ratioDraws, valueDraw, ratiosDecay, count)
	*ratioWins = movingAverage(*ratioWins, valueWin, ratiosDecay, count)
	*ratioLosses = movingAverage(*ratioLosses, valueLose, ratiosDecay, count)
}

func continuouslyRescoreWithEndScore(ctx context.Context, matchesIn <-chan *Match, matchesOut chan<- *Match, rescorePlayers ...PlayerNum) {
	defer func() {
		close(matchesOut)
	}()
	weight := float32(max(*flagTrainWithEndScore, 1))
	for match := range generics.IterChanWithContext(ctx, matchesIn) {
		winner := match.FinalBoard().Winner()
		var endScores [2]float32 // Scores for PlayerFirst and PlayerSecond.
		switch winner {
		case PlayerFirst:
			endScores = [2]float32{ai.WinGameScore, -ai.WinGameScore}
		case PlayerSecond:
			endScores = [2]float32{-ai.WinGameScore, ai.WinGameScore}
		default:
			// Draw: leave scores at 0.
		}
		for idx := range match.Actions {
			board := match.Boards[idx]
			endScore := endScores[board.NextPlayer]
			match.Scores[idx] = weight*endScore + (1-weight)*match.Scores[idx]
		}

		// Yield rescored match.
		select {
		case <-ctx.Done():
			klog.Errorf("End-game rescoring interrupted (context cancelled): %v", ctx.Err())
			return
		case matchesOut <- match:
			// Done
		}
	}
}
