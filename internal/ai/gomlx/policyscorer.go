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

// PolicyScorer implements a generic GoMLX "board scorer" (to use with AlphaBetaPruning or MinMax seachers) for the Hive game.
// It only models the estimate of the state value (Q).
//
// It implements ai.PolicyScorer, ai.BatchPolicyScorer and ai.ValueLearner.
//
// It is just a wrapper around on of the models implemented.
type PolicyScorer struct {
	Type ModelType

	// model if PolicyScorer is a a PolicyScorer.
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
		model:                 model,
		numPolicyInputTensors: -1,
		numLabelTensors:       -1,
	}

	// Help if requested.
	if slices.Index([]string{"help", "--help", "-help", "-h"}, filePath) != -1 {
		s.writeHyperparametersHelp()
		return nil, fmt.Errorf("model type %s help requested", modelType)
	}

	// Number of checkpoints to keep.
	var err error
	s.checkpointsToKeep, err = parameters.PopParamOr(params, "keep", 10)
	if err != nil {
		return nil, err
	}

	// Create checkpoint, and load it if it exists.
	if filePath != "" {
		if err = s.createCheckpoint(filePath); err != nil {
			return nil, errors.WithMessagef(err, "failed to build checkpoint for model %s in path %s",
				modelType, filePath)
		}
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

	// Setup scoreExec executor.
	muNewClient.Lock()
	defer muNewClient.Unlock()
	ctx = ctx.Checked(false)
	s.valueScoreExec = context.NewExec(backend(), ctx,
		func(ctx *context.Context, valueInputs []*graph.Node) *graph.Node {
			// Reshape to a scalar.
			return graph.Reshape(s.model.ForwardValueGraph(ctx, valueInputs))
		})
	s.policyScoreExec = context.NewExec(backend(), ctx,
		func(ctx *context.Context, policyInputs []*graph.Node) []*graph.Node {
			value, policy := s.model.ForwardPolicyGraph(ctx, policyInputs)
			return []*graph.Node{value, policy}
		})
	s.lossExec = context.NewExec(backend(), ctx,
		func(ctx *context.Context, inputsAndLabels []*graph.Node) *graph.Node {
			inputs := inputsAndLabels[:s.numPolicyInputTensors]
			labels := inputsAndLabels[s.numPolicyInputTensors:]
			loss := s.model.LossGraph(ctx, inputs, labels)
			if !loss.IsScalar() {
				// Some losses may return one value per example of the batch.
				loss = graph.ReduceAllMean(loss)
			}
			return loss
		})
	s.trainStepExec = context.NewExec(backend(), s.model.Context(),
		func(ctx *context.Context, inputsAndLabels []*graph.Node) *graph.Node {
			g := inputsAndLabels[0].Graph()
			inputs := inputsAndLabels[:s.numPolicyInputTensors]
			labels := inputsAndLabels[s.numPolicyInputTensors:]
			ctx.SetTraining(g, true)
			loss := s.model.LossGraph(ctx, inputs, labels)
			s.optimizer.UpdateGraph(ctx, g, loss)
			train.ExecPerStepUpdateGraphFn(ctx, g)
			return loss
		})

	// Force creating/loading of variables without race conditions first.
	board := state.NewBoard()
	_ = s.Score(board)

	return s, nil
}

// CloneLearner implements ai.PolicyLearner.
func (s *PolicyScorer) CloneLearner() ai.PolicyLearner {
	s.muLearning.Lock()
	defer s.muLearning.Unlock()
	s.muSave.Lock()
	defer s.muSave.Unlock()

	newS := &PolicyScorer{
		Type:                  s.Type,
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
	}
	// TODO: Set checkpoint to the same as s.
	// TODO: Create executors

	return newS
}

// String implements fmt.Stringer and ai.PolicyScorer.
func (s *PolicyScorer) String() string {
	if s == nil {
		return "<nil>[GoMLX]"
	}
	if s.checkpoint == nil {
		return fmt.Sprintf("%s[GoMLX]", s.Type)
	}
	return fmt.Sprintf("%s[GoMLX]@%s", s.Type, s.checkpoint.Dir())
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
// It returns the lossExec.
func (s *PolicyScorer) Learn(boards []*state.Board, valueLabels []float32, policyLabels [][]float32) (loss float32) {
	//fmt.Printf("Learn(%d boards)\n", len(boards))
	inputsAndLabels := s.createInputsAndLabels(boards, valueLabels, policyLabels)
	s.muLearning.Lock()
	defer s.muLearning.Unlock()
	lossT := s.trainStepExec.Call(inputsAndLabels...)[0]
	return tensors.ToScalar[float32](lossT)
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
	s.model.Context().EnumerateParams(func(scope, key string, value any) {
		if scope != context.RootScope {
			return
		}
		_, _ = fmt.Fprintf(buf, "\t%q: default value is %v\n", key, value)
	})
	klog.Info(buf)
}

func (s *PolicyScorer) createCheckpoint(filePath string) error {
	var err error
	s.checkpoint, err = checkpoints.
		Build(s.model.Context()).
		Dir(filePath).
		Immediate().
		Keep(10).
		Done()
	return err
}
