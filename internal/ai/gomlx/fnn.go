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
	return fnn
}

func (fnn *FNN) Context() *context.Context {
	return fnn.ctx
}

func (fnn *FNN) CreateInputs(boards []*state.Board) []*tensors.Tensor {
	version := context.GetParamOr(fnn.ctx, "features_version", features.BoardFeaturesDim)
	boardFeatures := tensors.FromShape(shapes.Make(dtypes.Float32, len(boards), version))
	tensors.MutableFlatData(boardFeatures, func(flat []float32) {
		for boardIdx, board := range boards {
			values := features.ForBoard(board, version)
			copy(flat[boardIdx*version:], values)
		}
	})
	return []*tensors.Tensor{boardFeatures}
}

// ForwardGraph calculates the scores of the board.
func (fnn *FNN) ForwardGraph(ctx *context.Context, inputs []*Node) *Node {
	return nil
}

// LossGraph calculates the lossExec.
func (fnn *FNN) LossGraph(ctx *context.Context, inputs []*Node, labels *Node) *Node {
	predictions := fnn.ForwardGraph(ctx, inputs)
	return losses.MeanSquaredError([]*Node{labels}, []*Node{predictions})
}
