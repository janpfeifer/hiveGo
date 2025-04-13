package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"math/rand"
	"strings"
	"time"
)

var (
	flagAIConfig = flag.String("ai", "", "Configuration for model/searcher to train. "+
		"a0trainer plays against itself, so only one configuration is accepted.")
	flagBootstrapAI = flag.String("bootstrap", "", "Configure an AI to bootstrap an AlphaZero model. "+
		"This is needed because randomly initialized models will never reach a win condition, and never learn anything.")
	flagTrainStepsPerIteration = flag.Int("train_steps", 1000, "Number of training steps to perform per iteration.")

	// aiPlayer being trained.
	aiPlayer          *players.SearcherScorer
	bootstrapAiPlayer *players.SearcherScorer
)

// Example hold one data point to learn from.
type Example struct {
	board        *state.Board
	valueLabel   float32
	policyLabels []float32
}

func createAIPlayer() error {
	config := *flagAIConfig
	if config == "" {
		return errors.New("must specify AI configuration with -ai")
	}
	if strings.Index(config, ";") != -1 {
		return errors.Errorf("invalid AI config %q, only one AI configuration must be given, no \";\" accepted", config)
	}
	klog.V(1).Infof("Creating AI from %q", config)
	var err error
	aiPlayer, err = players.New(config)
	if err != nil {
		return err
	}
	if aiPlayer.PolicyLearner == nil {
		return errors.Errorf("invalid AI config (-ai): a0trainer requires a \"PolicyLearner\" model, "+
			"but %s doesn't seem to implement it", aiPlayer.ValueScorer)
	}
	if *flagBootstrapAI != "" {
		bootstrapAiPlayer, err = players.New(*flagBootstrapAI)
		if err != nil {
			return errors.WithMessagef(err, "invalid bootstrap AI config (-bootstrap-ai) %q", *flagBootstrapAI)
		}
	}
	return nil
}

func trainAI(ctx context.Context, examples []Example) (success bool, err error) {
	learner := aiPlayer.PolicyLearner
	batchSize := learner.BatchSize()
	var averageLoss float32

	var numSteps int
	start := time.Now()
	printUpdate := func() {
		elapsed := time.Since(start)
		fmt.Printf("\r\tTraining: %d steps, ~loss=%.3f, elpased=%s\x1b[0K", numSteps, averageLoss, elapsed)
	}
	printUpdate()

	numLearnSteps := *flagTrainStepsPerIteration
	err = exceptions.TryCatch[error](func() {
		boardsBatch := make([]*state.Board, batchSize)
		valueLabelsBatch := make([]float32, batchSize)
		policyLabelsBatch := make([][]float32, batchSize)
		for range numLearnSteps {
			if ctx.Err() != nil {
				return
			}
			// Sample batch: random with replacement:
			for batchIdx := range batchSize {
				example := examples[rand.Intn(len(examples))]
				boardsBatch[batchIdx] = example.board
				valueLabelsBatch[batchIdx] = example.valueLabel
				policyLabelsBatch[batchIdx] = example.policyLabels
			}
			loss := learner.Learn(boardsBatch, valueLabelsBatch, policyLabelsBatch)
			numSteps++
			averageLoss = movingAverage(averageLoss, loss, averageLossDecay, numSteps)
			printUpdate()
		}
	})
	fmt.Println()
	if err != nil {
		return false, err
	}
	if ctx.Err() != nil {
		// Interrupted.
		return false, nil
	}

	// TODO: check trained model is better than previous one.

	// Save model.
	err = learner.Save()
	if err != nil {
		return false, errors.WithMessagef(err, "failed to save model after training")
	}

	// Success
	return true, nil
}

const averageLossDecay = float32(0.95)

func movingAverage(average, newValue, decay float32, count int) float32 {
	decay = min(1-1/float32(count), decay)
	return average*decay + (1-decay)*newValue
}
