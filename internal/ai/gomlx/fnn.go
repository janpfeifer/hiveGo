package gomlx

import (
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/ml/train/optimizers/cosineschedule"
)

// FNN implement a feed-forward model on the board features.
// It's the simpler GoMLX model.
type FNN struct {
}

// CreateContext creates a context with the default hyperparameters set.
func (f *FNN) CreateContext() *context.Context {
	ctx := context.New()
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		"batch_size": 128,

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
		fnn.ParamNumHiddenLayers: 1,
		fnn.ParamNumHiddenNodes:  4,
		fnn.ParamResidual:        true,
		fnn.ParamNormalization:   "layer",

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
	return ctx
}
