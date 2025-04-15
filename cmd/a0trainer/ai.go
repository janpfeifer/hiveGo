package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/ai/gomlx"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/searchers/mcts"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"math/rand"
	"runtime"
	"strings"
	"time"
)

var (
	flagAIConfig = flag.String("ai", "", "Configuration for model/searcher to train. "+
		"a0trainer plays against itself, so only one configuration is accepted.")
	flagBootstrapAI = flag.String("bootstrap", "", "Configure an AI to bootstrap an AlphaZero model. "+
		"This is needed because randomly initialized models will never reach a win condition, and never learn anything.")
	flagTrainStepsPerExample = flag.Int("train_steps", 10, "Average number of times that each example (board move) is seen during a training round.")

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
	if *flagTrainStepsPerExample <= 0 {
		// No training, probably just a debug run.
		return true, nil
	}
	learner := aiPlayer.PolicyLearner
	batchSize := learner.BatchSize()
	numTrainSteps := *flagTrainStepsPerExample * len(examples) / batchSize
	if numTrainSteps <= 0 {
		// Not enough examples for even one batch.
		return false, nil
	}
	fmt.Printf("\t- Training %d steps, batch size %d, pool of %d examples\n", numTrainSteps, batchSize, len(examples))
	var averageLoss float32

	// Clone learner to a new one, and clear its optimizer before start training.
	newLearner, err := learner.CloneLearner()
	if err != nil {
		return false, errors.WithMessagef(err, "failed to clone learner")
	}
	newLearner.ClearOptimizer()

	// Stats of training.
	var currentStep int
	start := time.Now()
	printUpdate := func() {
		elapsed := time.Since(start)
		fmt.Printf("\r\tTraining: %6d steps, ~loss=%.3f, elpased=%s\x1b[0K", currentStep, averageLoss, elapsed)
	}
	printUpdate()

	err = exceptions.TryCatch[error](func() {
		boardsBatch := make([]*state.Board, batchSize)
		valueLabelsBatch := make([]float32, batchSize)
		policyLabelsBatch := make([][]float32, batchSize)
		for range numTrainSteps {
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
			loss := newLearner.Learn(boardsBatch, valueLabelsBatch, policyLabelsBatch)
			currentStep++
			averageLoss = movingAverage(averageLoss, loss, averageLossDecay, currentStep)
			printUpdate()
		}
	})
	printUpdate()
	fmt.Println()
	if err != nil {
		return false, err
	}
	if ctx.Err() != nil {
		// Interrupted.
		return false, nil
	}
	if policyScorer, ok := newLearner.(*gomlx.PolicyScorer); ok {
		fmt.Printf("\t- Number of cached compiled graphs (for different shapes): %d\n", policyScorer.NumCompilations)
	}

	// Create newPlayer: it requires re-creating the Searcher with the new scorer.
	newPlayer := &players.SearcherScorer{
		ValueScorer:   newLearner.(ai.ValueScorer),
		ValueLearner:  nil,
		PolicyScorer:  newLearner.(ai.PolicyScorer),
		PolicyLearner: newLearner,
	}
	mctsSearcher, ok := aiPlayer.PolicySearcher.(*mcts.Searcher)
	if !ok {
		return false, errors.Errorf("invalid AI config (-ai): a0trainer requires a \"mcts.Searcher\" searcher, " +
			"but the given searcher is of a different type.")
	}
	newSearcher := mctsSearcher.Clone().WithScorer(newPlayer.PolicyScorer)
	newPlayer.PolicySearcher = newSearcher
	newPlayer.Searcher = newSearcher

	// Check whether new model is better than previous one.
	if *flagNumCompareMatches > 0 {
		currentWins, newWins, draws, _, err := runMatches(ctx, *flagNumCompareMatches, aiPlayer, newPlayer)
		if err != nil {
			err = errors.WithMessagef(err, "failed to run matches to compare models after training")
			return false, err
		}
		if ctx.Err() != nil {
			// Interrupted.
			return false, nil
		}
		fmt.Printf("\t- %d draws, %d current model wins, %d updated model wins\n", draws, currentWins, newWins)
		if newWins <= currentWins+(currentWins+9)/10 {
			// Didn't win at least >10% more than current model, discard training and instead collect more examples.
			fmt.Printf("\t- Discarding training, not enough wins to be worth it. Collecting more examples.\n")
			if policyScorer, ok := newLearner.(*gomlx.PolicyScorer); ok {
				policyScorer.Finalize()
			}
			newLearner = nil
			newPlayer = nil
			for _ = range 5 {
				runtime.GC()
			}
			return false, nil
		}
	}

	// Free old model.
	if policyScorer, ok := learner.(*gomlx.PolicyScorer); ok {
		policyScorer.Finalize()
	}
	aiPlayer = nil
	for _ = range 5 {
		runtime.GC()
	}

	// Take update model and save it.
	aiPlayer = newPlayer
	fmt.Printf("\t- New model is better than current model, replacing and saving it.\n")
	err = newLearner.Save()
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
