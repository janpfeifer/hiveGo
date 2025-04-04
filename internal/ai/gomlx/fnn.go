package gomlx

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	fnnLayer "github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/hiveGo/internal/features"
	"github.com/janpfeifer/hiveGo/internal/state"
)

// FNN implement a feed-forward model on the board features.
// It's the simpler GoMLX model.
type FNN struct {
	ctx *context.Context
}

// NewFNN creates an FNN model with a fresh context, initialized with hyperparameters set to their defaults.
func NewFNN() *FNN {
	fnn := &FNN{ctx: context.New()}
	fnn.ctx.RngStateReset()
	fnn.ctx.SetParams(map[string]any{
		"batch_size": 128,

		// Number of board features to extract.
		// This allows backward compatibility, otherwise better leave as is.
		"features_version": features.BoardFeaturesDim,

		optimizers.ParamOptimizer:       "adam",
		optimizers.ParamLearningRate:    0.001,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamDType:       "",
		cosineschedule.ParamPeriodSteps: 0,
		activations.ParamActivation:     "sigmoid",
		layers.ParamDropoutRate:         0.0,
		regularizers.ParamL2:            1e-5,
		regularizers.ParamL1:            1e-5,

		// FNN network parameters:
		fnnLayer.ParamNumHiddenLayers: 1,
		fnnLayer.ParamNumHiddenNodes:  4,
		fnnLayer.ParamResidual:        true,
		fnnLayer.ParamNormalization:   "layer",

		// KAN network parameters:
		"kan":                                 false, // Enable kan
		kan.ParamNumControlPoints:             20,    // Number of control points
		kan.ParamNumHiddenNodes:               4,
		kan.ParamNumHiddenLayers:              1,
		kan.ParamBSplineDegree:                2,
		kan.ParamBSplineMagnitudeL1:           1e-5,
		kan.ParamBSplineMagnitudeL2:           0.0,
		kan.ParamDiscrete:                     false,
		kan.ParamDiscretePerturbation:         "triangular",
		kan.ParamDiscreteSoftness:             0.1,
		kan.ParamDiscreteSoftnessSchedule:     kan.SoftnessScheduleNone.String(),
		kan.ParamDiscreteSplitPointsTrainable: true,
		kan.ParamResidual:                     true,
	})
	fnn.ctx = fnn.ctx.Checked(false)
	return fnn
}

func (fnn *FNN) Context() *context.Context {
	return fnn.ctx
}

// paddedBatchSize returns a padded batchSize for the given numBoards.
// This is important so we don't have too many different versions of the program for every different batch size.
func (fnn *FNN) paddedBatchSize(numBoards int) int {
	// Make sure the default batchSize is supported without padding.
	defaultBatchSize := context.GetParamOr(fnn.ctx, "batch_size", 128)
	if numBoards == defaultBatchSize {
		return numBoards
	}

	paddedSize := 1
	for paddedSize < numBoards {
		// Increase 1.5x at a time.
		paddedSize = paddedSize + (paddedSize+1)/2
	}
	return paddedSize
}

// CreateInputs implements ValueModel.CreateInputs.
func (fnn *FNN) CreateInputs(boards []*state.Board) []*tensors.Tensor {
	version := context.GetParamOr(fnn.ctx, "features_version", features.BoardFeaturesDim)
	numBoards := len(boards)
	paddedBatchSize := fnn.paddedBatchSize(numBoards)
	boardFeatures := tensors.FromShape(shapes.Make(dtypes.Float32, paddedBatchSize, version))
	tensors.MutableFlatData(boardFeatures, func(flat []float32) {
		for boardIdx, board := range boards {
			values := features.ForBoard(board, version)
			copy(flat[boardIdx*version:], values)
		}
	})
	return []*tensors.Tensor{boardFeatures, tensors.FromScalar(int32(len(boards)))}
}

// CreateLabels implements ValueModel.CreateLabels.
func (fnn *FNN) CreateLabels(labels []float32) *tensors.Tensor {
	paddedBatchSize := fnn.paddedBatchSize(len(labels))
	boardLabels := tensors.FromShape(shapes.Make(dtypes.Float32, paddedBatchSize, 1))
	tensors.MutableFlatData(boardLabels, func(flat []float32) {
		copy(flat, labels)
	})
	return boardLabels
}

// getBatchMask based on padding on the inputs.
func (fnn *FNN) getBatchMask(inputs []*Node) *Node {
	logits := inputs[0]
	usedBatchSize := inputs[1]
	g := logits.Graph()
	batchSize := logits.Shape().Dim(0)
	batchMask := LessThan(Iota(g, shapes.Make(dtypes.Int32, batchSize, 1), 0), usedBatchSize)
	return batchMask
}

// ForwardGraph calculates the scores of the board.
func (fnn *FNN) ForwardGraph(ctx *context.Context, inputs []*Node) *Node {
	logits := inputs[0]
	batchSize := logits.Shape().Dim(0)

	// ValueModel itself is an FNN or a KAN.
	if context.GetParamOr(ctx, "kan", false) {
		// Use KAN, all configured by context hyperparameters. See createDefaultContext for defaults.
		logits = kan.New(ctx.In("kan"), logits, 1).Done()
	} else {
		// Normal FNN, all configured by context hyperparameters. See createDefaultContext for defaults.
		logits = fnnLayer.New(ctx.In("fnn"), logits, 1).Done()
	}
	logits.AssertDims(batchSize, 1) // 2-dim tensor, with batch size as the leading dimension.
	predictions := MulScalar(Tanh(logits), 0.99)
	return predictions
}

// LossGraph calculates the lossExec.
func (fnn *FNN) LossGraph(ctx *context.Context, inputs []*Node, labels *Node) *Node {
	predictions := fnn.ForwardGraph(ctx, inputs)
	batchMask := fnn.getBatchMask(inputs)
	return losses.MeanSquaredError([]*Node{labels, batchMask}, []*Node{predictions})
}
