package gomlx

import (
	"bytes"
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// newBoardScorer returns a gomlx.BoardScorer for the given ValueModel.
func newBoardScorer(modelType ModelType, filePath string, model ValueModel, params parameters.Params) (*BoardScorer, error) {
	boardScorer := &BoardScorer{
		Type:  modelType,
		model: model,
	}

	// Help if requested.
	if slices.Index([]string{"help", "--help", "-help", "-h"}, filePath) != -1 {
		boardScorer.writeHyperparametersHelp()
		return nil, fmt.Errorf("model type %s help requested", modelType)
	}

	// Number of checkpoints to keep.
	var err error
	boardScorer.checkpointsToKeep, err = parameters.PopParamOr(params, "keep", 10)
	if err != nil {
		return nil, err
	}

	// Create checkpoint, and load it if it exists.
	if filePath != "" {
		if err = boardScorer.createCheckpoint(filePath); err != nil {
			return nil, errors.WithMessagef(err, "failed to build checkpoint for model %s in path %s",
				modelType, filePath)
		}
	}

	// Create the backend.
	_ = backend()

	// Overwrite hyperparameters from given params.
	err = extractParams(boardScorer.Type.String(), params, boardScorer.model.Context())
	if err != nil {
		return nil, err
	}
	ctx := boardScorer.model.Context()
	boardScorer.batchSize = context.GetParamOr(ctx, "batch_size", 100)

	// Create optimizer to be used in training.
	boardScorer.optimizer = optimizers.FromContext(ctx)

	// Setup scoreExec executor.
	muNewClient.Lock()
	defer muNewClient.Unlock()
	boardScorer.scoreExec = context.NewExec(backend(), ctx,
		func(ctx *context.Context, inputs []*graph.Node) *graph.Node {
			// Remove last axis with dimension 1.
			ctx = ctx.Checked(false)
			return graph.Squeeze(boardScorer.model.ForwardGraph(ctx, inputs), -1)
		})
	boardScorer.lossExec = context.NewExec(backend(), ctx,
		func(ctx *context.Context, inputsAndLabels []*graph.Node) *graph.Node {
			inputs := inputsAndLabels[:len(inputsAndLabels)-1]
			labels := inputsAndLabels[len(inputsAndLabels)-1]
			if labels.Rank() == 1 {
				// Add the last axes with dimension 1.
				labels = graph.ExpandAxes(labels, -1)
			}
			loss := boardScorer.model.LossGraph(ctx, inputs, labels)
			if !loss.IsScalar() {
				// Some losses may return one value per example of the batch.
				loss = graph.ReduceAllMean(loss)
			}
			return loss
		})
	boardScorer.trainStepExec = context.NewExec(backend(), boardScorer.model.Context(),
		func(ctx *context.Context, inputsAndLabels []*graph.Node) *graph.Node {
			inputs := inputsAndLabels[:len(inputsAndLabels)-1]
			labels := inputsAndLabels[len(inputsAndLabels)-1]
			g := labels.Graph()
			ctx.SetTraining(g, true)
			loss := boardScorer.model.LossGraph(ctx, inputs, labels)
			boardScorer.optimizer.UpdateGraph(ctx, g, loss)
			train.ExecPerStepUpdateGraphFn(ctx, g)
			return loss
		})

	// Force creating/loading of variables without race conditions first.
	board := state.NewBoard()
	_ = boardScorer.Score(board)

	return boardScorer, nil
}

// BoardScorer implements a generic GoMLX "board scorer" (to use with AlphaBetaPruning or MinMax seachers) for the Hive game.
// It only models the estimate of the state value (Q).
//
// It implements ai.ValueScorer, ai.BatchValueScorer and ai.ValueLearner.
//
// It is just a wrapper around on of the models implemented.
type BoardScorer struct {
	Type ModelType

	// model if BoardScorer is a a BoardScorer.
	model ValueModel

	// Executors.
	scoreExec, lossExec, trainStepExec *context.Exec

	// checkpoint handler, if model is being saved/loaded to/from disk.
	checkpoint *checkpoints.Handler

	// checkpointsToKeep is the number of copies of older checkpoints to keep around.
	// Default to 10.
	checkpointsToKeep int

	// Hyperparameters cached values: they should also be set in modelCtx.
	batchSize int

	// muLearning "write" for learning, and "read" for scoring.
	muLearning sync.RWMutex

	// optimizer used when training the model.
	// ?Should this be owned by the model itself?
	optimizer optimizers.Interface

	// muSave makes saving sequential.
	muSave sync.Mutex
}

var (
	// Assert BoardScorer is an ai.ValueScorer, an ai.BatchValueScorer and an ai.ValueLearner.
	_ ai.ValueScorer      = (*BoardScorer)(nil)
	_ ai.BatchValueScorer = (*BoardScorer)(nil)
	_ ai.ValueLearner     = (*BoardScorer)(nil)
)

// String implements fmt.Stringer and ai.ValueScorer.
func (s *BoardScorer) String() string {
	if s == nil {
		return "<nil>[GoMLX]"
	}
	if s.checkpoint == nil {
		return fmt.Sprintf("%s[GoMLX]", s.Type)
	}
	return fmt.Sprintf("%s[GoMLX]@%s", s.Type, s.checkpoint.Dir())
}

// BoardScore implements ai.ValueScorer.
func (s *BoardScorer) Score(board *state.Board) float32 {
	return s.BatchScore([]*state.Board{board})[0]
}

// BatchBoardScore implements ai.BatchValueScorer.
func (s *BoardScorer) BatchScore(boards []*state.Board) []float32 {
	inputs := s.model.CreateInputs(boards)

	s.muLearning.RLock()
	defer s.muLearning.RUnlock()
	donatedInputs := generics.SliceMap(inputs, func(t *tensors.Tensor) any {
		return graph.DonateTensorBuffer(t, backend())
	})

	scoresT := s.scoreExec.Call(donatedInputs...)[0]
	scores := scoresT.Value().([]float32)
	// Remove any padding:
	return scores[:len(boards)]
}

// Learn implements ai.ValueLearner, and trains model with the new boards and its labels.
// It returns the lossExec.
func (s *BoardScorer) Learn(boards []*state.Board, boardLabels []float32) (loss float32) {
	//fmt.Printf("Learn(%d boards)\n", len(boards))
	s.muLearning.Lock()
	defer s.muLearning.Unlock()
	lossT := s.trainStepExec.Call(s.createInputsAndLabels(boards, boardLabels)...)[0]
	return tensors.ToScalar[float32](lossT)
}

// Loss returns a measure of lossExec for the model -- whatever it is.
func (s *BoardScorer) Loss(boards []*state.Board, boardLabels []float32) (loss float32) {
	s.muLearning.RLock()
	defer s.muLearning.RUnlock()
	lossT := s.lossExec.Call(s.createInputsAndLabels(boards, boardLabels)...)[0]
	return tensors.ToScalar[float32](lossT)
}

func (s *BoardScorer) createInputsAndLabels(boards []*state.Board, boardLabels []float32) []any {
	inputs := s.model.CreateInputs(boards)
	inputs = append(inputs, s.model.CreateLabels(boardLabels))
	donatedInputs := generics.SliceMap(inputs, func(t *tensors.Tensor) any {
		return graph.DonateTensorBuffer(t, backend())
	})
	return donatedInputs
}

// Save should save the model.
func (s *BoardScorer) Save() error {
	if s.checkpoint == nil {
		klog.Warningf("This %s model is not associated to a checkpoint directory,  not saving", s.Type)
		return nil
	}
	return s.checkpoint.Save()
}

// BatchSize returns the recommended batch size and implements ai.ValueLearner.
func (s *BoardScorer) BatchSize() int {
	return s.batchSize
}

// writeHyperparametersHelp enumerates all the hyperparameters set in the context.
func (s *BoardScorer) writeHyperparametersHelp() {
	buf := &bytes.Buffer{}
	_, _ = fmt.Fprintf(buf, "ValueModel %s parameters:\n", s.Type)
	_, _ = fmt.Fprintf(buf, "\tfnn=<path_to_model> to use the model saved at the given directory, or\n")
	_, _ = fmt.Fprintf(buf, "\tfnn=#0 to use pretrained model number 0 (there are %d pretrained models) or\n", len(PretrainedModels[ModelFNN]))
	_, _ = fmt.Fprintf(buf, "\tfnn=-help to show this help message\n")
	s.model.Context().EnumerateParams(func(scope, key string, value any) {
		if scope != context.RootScope {
			return
		}
		_, _ = fmt.Fprintf(buf, "\t%q: default value is %v\n", key, value)
	})
	klog.Info(buf)
}

func (s *BoardScorer) createCheckpoint(filePath string) error {
	checkpoint, err := genericCreateCheckpoint(s.model.Context(), ModelFNN, filePath)
	if err != nil {
		return err
	}
	s.checkpoint = checkpoint
	return nil
}

func genericCreateCheckpoint(ctx *context.Context, modelType ModelType, filePath string) (*checkpoints.Handler, error) {
	checkpointConfig := checkpoints.
		Build(ctx).
		Immediate().
		Keep(10)

	if strings.HasPrefix(filePath, "#") {
		// Load pre-trained model.
		pretrainedIdx, err := strconv.Atoi(filePath[1:])
		if err != nil {
			return nil, errors.Wrapf(err, "failed to parse the pretrained model index from %q", filePath)
		}
		numPretrained := len(PretrainedModels[modelType])
		if pretrainedIdx < 0 || pretrainedIdx >= numPretrained {
			return nil, errors.Errorf("model type %s only have %d pretrained models included, please select a number from 0 to %d",
				modelType, numPretrained, numPretrained-1)
		}
		checkpointConfig = checkpointConfig.FromEmbed(
			PretrainedModels[modelType][pretrainedIdx].Json,
			PretrainedModels[modelType][pretrainedIdx].Binary)
	} else {
		// Load/Save model from/to disk.
		checkpointConfig = checkpointConfig.Dir(filePath)
	}
	checkpoint, err := checkpointConfig.Done()
	return checkpoint, err
}
