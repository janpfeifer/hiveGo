package gomlx

import (
	"bytes"
	"fmt"
	"github.com/gomlx/exceptions"
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
	"sync"
)

// PolicyScorer implements a generic GoMLX "board scorer" (to use with AlphaBetaPruning or MinMax searchers) for the Hive game.
// It only models the estimate of the state value (Q).
//
// It implements ai.PolicyScorer, ai.BatchPolicyScorer and ai.ValueLearner.
//
// It is just a wrapper around on of the models implemented.
type PolicyScorer struct {
	Type ModelType

	// filePath passed to the model, where it is saved.
	filePath string

	// model if PolicyScorer is a PolicyScorer.
	model PolicyModel

	// Executors.
	valueScoreExec, policyScoreExec, lossExec, trainStepExec *context.Exec

	// Number of input tensors for the executors: they are defined at the first call to
	// PolicyModel.CreatePolicyInputs and PolicyModel.CreatePolicyLabels, and must remain constant.
	// Before they are defined, they are temporarily set as -1.
	numPolicyInputTensors, numLabelTensors int

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

	// numCompilations of computation graphs.
	NumCompilations int

	// muSave makes saving sequential.
	muSave sync.Mutex
}

var (
	// Assert PolicyScorer is an ai.PolicyScorer, an ai.BatchPolicyScorer and an ai.ValueLearner.
	_ ai.PolicyScorer  = (*PolicyScorer)(nil)
	_ ai.PolicyLearner = (*PolicyScorer)(nil)
)

// newPolicyScorer returns a gomlx.PolicyScorer for the given ValueModel.
func newPolicyScorer(modelType ModelType, filePath string, model PolicyModel, params parameters.Params) (*PolicyScorer, error) {
	s := &PolicyScorer{
		Type:                  modelType,
		filePath:              filePath,
		model:                 model,
		numPolicyInputTensors: -1,
		numLabelTensors:       -1,
	}

	// Help if requested.
	if slices.Index([]string{"help", "--help", "-help", "-h"}, filePath) != -1 {
		s.writeHyperparametersHelp()
		return nil, fmt.Errorf("model type %s help requested", modelType)
	}

	// Checkpoint model.
	var err error
	s.checkpointsToKeep, err = parameters.PopParamOr(params, "keep", 10)
	if err != nil {
		return nil, err
	}
	err = s.connectCheckpointHandler()
	if err != nil {
		return nil, err
	}

	// Create the backend.
	_ = backend()

	// Overwrite hyperparameters from given params.
	err = extractParams(s.Type.String(), params, s.model.Context())
	if err != nil {
		return nil, err
	}
	ctx := s.model.Context()
	s.batchSize = context.GetParamOr(ctx, "batch_size", 100)

	// Create optimizer to be used in training.
	s.optimizer = optimizers.FromContext(ctx)
	s.createExecutors()
	return s, nil
}

func (s *PolicyScorer) connectCheckpointHandler() error {
	if s.filePath == "" {
		return nil
	}
	if err := s.createCheckpoint(s.filePath); err != nil {
		return errors.WithMessagef(err, "failed to build checkpoint for model %s in path %s",
			s.Type, s.filePath)
	}
	return nil
}

func (s *PolicyScorer) createExecutors() {
	muNewClient.Lock()
	defer muNewClient.Unlock()
	ctx := s.model.Context().Checked(false)
	s.valueScoreExec = context.NewExec(backend(), ctx,
		func(ctx *context.Context, valueInputs []*graph.Node) *graph.Node {
			// Reshape to a scalar.
			s.NumCompilations++
			return graph.Reshape(s.model.ForwardValueGraph(ctx, valueInputs))
		})
	s.policyScoreExec = context.NewExec(backend(), ctx,
		func(ctx *context.Context, policyInputs []*graph.Node) []*graph.Node {
			s.NumCompilations++
			value, policy := s.model.ForwardPolicyGraph(ctx, policyInputs)
			return []*graph.Node{value, policy}
		})
	s.lossExec = context.NewExec(backend(), ctx,
		func(ctx *context.Context, inputsAndLabels []*graph.Node) *graph.Node {
			s.NumCompilations++
			inputs := inputsAndLabels[:s.numPolicyInputTensors]
			labels := inputsAndLabels[s.numPolicyInputTensors:]
			loss := s.model.LossGraph(ctx, inputs, labels)
			if !loss.IsScalar() {
				// Some losses may return one value per example of the batch.
				loss = graph.ReduceAllMean(loss)
			}
			return loss
		})
	s.lossExec.SetMaxCache(100)
	s.trainStepExec = context.NewExec(backend(), s.model.Context(),
		func(ctx *context.Context, inputsAndLabels []*graph.Node) *graph.Node {
			s.NumCompilations++
			g := inputsAndLabels[0].Graph()
			ctx.SetTraining(g, true)
			inputs := inputsAndLabels[:s.numPolicyInputTensors]
			labels := inputsAndLabels[s.numPolicyInputTensors:]
			loss := s.model.LossGraph(ctx, inputs, labels)
			s.optimizer.UpdateGraph(ctx, g, loss)
			train.ExecPerStepUpdateGraphFn(ctx, g)
			return loss
		})
	s.trainStepExec.SetMaxCache(100)

	// Force creating/loading of variables without race conditions first.
	board := state.NewBoard()
	_ = s.PolicyScore(board)
	_ = s.Score(board)
}

// CloneLearner implements ai.PolicyLearner.
func (s *PolicyScorer) CloneLearner() (ai.PolicyLearner, error) {
	s.muLearning.Lock()
	defer s.muLearning.Unlock()
	s.muSave.Lock()
	defer s.muSave.Unlock()

	newS := &PolicyScorer{
		Type:                  s.Type,
		filePath:              s.filePath,
		model:                 s.model.Clone(),
		valueScoreExec:        nil,
		policyScoreExec:       nil,
		lossExec:              nil,
		trainStepExec:         nil,
		numPolicyInputTensors: s.numPolicyInputTensors,
		numLabelTensors:       s.numLabelTensors,
		checkpoint:            nil,
		checkpointsToKeep:     s.checkpointsToKeep,
		batchSize:             s.batchSize,
		optimizer:             s.optimizer,
		NumCompilations:       0, // New model has no compiled computations graphs yet.
	}

	newS.createExecutors()
	err := newS.connectCheckpointHandler()
	if err != nil {
		return nil, err
	}
	return newS, nil
}

// String implements fmt.Stringer and ai.PolicyScorer.
func (s *PolicyScorer) String() string {
	if s == nil {
		return "<nil>[GoMLX]"
	}
	gomlxName := fmt.Sprintf("[GoMLX/%s]", backend().Name())
	if s.checkpoint == nil || s.checkpoint.Dir() == "" {
		return fmt.Sprintf("%s%s", s.Type, gomlxName)
	}

	return fmt.Sprintf("%s%s@%s", s.Type, gomlxName, s.checkpoint.Dir())
}

// Score implements ai.PolicyScorer (which includes ai.ValueScorer).
func (s *PolicyScorer) Score(board *state.Board) float32 {
	inputs := s.model.CreateValueInputs(board)
	s.muLearning.RLock()
	defer s.muLearning.RUnlock()
	donatedInputs := generics.SliceMap(inputs, func(t *tensors.Tensor) any {
		return graph.DonateTensorBuffer(t, backend())
	})

	scoreT := s.valueScoreExec.Call(donatedInputs...)[0]
	return tensors.ToScalar[float32](scoreT)
}

// BatchScore implements ai.BatchPolicyScorer.
func (s *PolicyScorer) BatchScore(boards []*state.Board) []float32 {
	return generics.SliceMap(boards, func(board *state.Board) float32 {
		return s.Score(board)
	})
}

// createPolicyInputs is a wrapper over s.model.CreatePolicyInputs that asserts the number of inputs hasn't changed.
func (s *PolicyScorer) createPolicyInputs(boards []*state.Board) []*tensors.Tensor {
	inputs := s.model.CreatePolicyInputs(boards)
	if s.numPolicyInputTensors == -1 {
		s.numPolicyInputTensors = len(inputs)
	} else {
		if len(inputs) != s.numPolicyInputTensors {
			exceptions.Panicf("model %s: expected %d policy inputs, got %d",
				s, s.numPolicyInputTensors, len(inputs))
		}
	}
	return inputs
}

// PolicyScore implements ai.PolicyScorer.
//
// It automatically trims the padding (if any) used by the PolicyModel used.
func (s *PolicyScorer) PolicyScore(board *state.Board) []float32 {
	inputs := s.createPolicyInputs([]*state.Board{board})
	s.muLearning.RLock()
	defer s.muLearning.RUnlock()
	donatedInputs := generics.SliceMap(inputs, func(t *tensors.Tensor) any {
		return graph.DonateTensorBuffer(t, backend())
	})
	policyScoresT := s.policyScoreExec.Call(donatedInputs...)[1]
	paddedPolicyScores := tensors.CopyFlatData[float32](policyScoresT)
	// Notice this works because we are scoring only one board, if it were a batch, we would need to deal with the
	// ragged actions tensor.
	return paddedPolicyScores[:board.NumActions()]
}

// Learn implements ai.ValueLearner, and trains model with the new boards and its labels.
//
// The input should be one batch, and this performs one "training step".
//
// It returns the lossExec.
func (s *PolicyScorer) Learn(boards []*state.Board, valueLabels []float32, policyLabels [][]float32) (loss float32) {
	//fmt.Printf("Learn(%d boards)\n", len(boards))
	inputsAndLabels := s.createInputsAndLabels(boards, valueLabels, policyLabels)
	s.muLearning.Lock()
	defer s.muLearning.Unlock()
	lossT := s.trainStepExec.Call(inputsAndLabels...)[0]
	return tensors.ToScalar[float32](lossT)
}

// ClearOptimizer variables and the global step.
func (s *PolicyScorer) ClearOptimizer() {
	s.muLearning.Lock()
	defer s.muLearning.Unlock()
	ctx := s.model.Context()
	optimizers.DeleteGlobalStep(ctx)
	s.optimizer.Clear(ctx)
}

// Loss returns a measure of lossExec for the model -- whatever it is.
func (s *PolicyScorer) Loss(boards []*state.Board, valueLabels []float32, policyLabels [][]float32) (loss float32) {
	inputsAndLabels := s.createInputsAndLabels(boards, valueLabels, policyLabels)
	s.muLearning.RLock()
	defer s.muLearning.RUnlock()
	lossT := s.lossExec.Call(inputsAndLabels...)[0]
	return tensors.ToScalar[float32](lossT)
}

func (s *PolicyScorer) createInputsAndLabels(boards []*state.Board, valueLabels []float32, policyLabels [][]float32) []any {
	inputs := s.createPolicyInputs(boards)
	labels := s.model.CreatePolicyLabels(valueLabels, policyLabels)
	if s.numLabelTensors == -1 {
		s.numLabelTensors = len(labels)
	} else {
		if len(labels) != s.numLabelTensors {
			exceptions.Panicf("model %s: expected %d policy label tensors, got %d", s, s.numLabelTensors, len(labels))
		}
	}
	inputs = append(inputs, labels...)
	donatedInputs := generics.SliceMap(inputs, func(t *tensors.Tensor) any {
		return graph.DonateTensorBuffer(t, backend())
	})
	return donatedInputs
}

// Save should save the model.
func (s *PolicyScorer) Save() error {
	if s.checkpoint == nil {
		klog.Warningf("This %s model is not associated to a checkpoint directory,  not saving", s.Type)
		return nil
	}
	return s.checkpoint.Save()
}

// BatchSize returns the recommended batch size and implements ai.ValueLearner.
func (s *PolicyScorer) BatchSize() int {
	return s.batchSize
}

// writeHyperparametersHelp enumerates all the hyperparameters set in the context.
func (s *PolicyScorer) writeHyperparametersHelp() {
	buf := &bytes.Buffer{}
	_, _ = fmt.Fprintf(buf, "ValueModel %s parameters:\n", s.Type)
	_, _ = fmt.Fprintf(buf, "\ta0fnn=<path_to_model> to use the model saved at the given directory, or\n")
	_, _ = fmt.Fprintf(buf, "\ta0fnn=#0 to use pretrained model number 0 (there are %d pretrained models) or\n", len(PretrainedModels[ModelAlphaZeroFNN]))
	_, _ = fmt.Fprintf(buf, "\ta0fnn=-help to show this help message\n")
	s.model.Context().EnumerateParams(func(scope, key string, value any) {
		if scope != context.RootScope {
			return
		}
		_, _ = fmt.Fprintf(buf, "\t%q: default value is %v\n", key, value)
	})

	klog.Info(buf)
}

func (s *PolicyScorer) createCheckpoint(filePath string) error {
	checkpoint, err := genericCreateCheckpoint(s.model.Context(), ModelAlphaZeroFNN, filePath)
	if err != nil {
		return err
	}
	s.checkpoint = checkpoint
	return nil
}

// Finalize associated model, and leaves scorer in an invalid state, but immediately frees resources.
func (s *PolicyScorer) Finalize() {
	s.valueScoreExec.Finalize()
	s.policyScoreExec.Finalize()
	s.lossExec.Finalize()
	s.trainStepExec.Finalize()
	s.model.Context().Finalize()
}
