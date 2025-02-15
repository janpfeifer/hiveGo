package linear

import (
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Embedded pre-trained linear models.

var (
	// PreTrainedV0 weights were trained with Scorer.Learn.
	PreTrainedV0 = NewWithWeights(
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// IdNumOffboard
		-0.43, 0.04, -0.52, -2.02, -0.64,
		// IdOpponentNumOffboard
		0.53, 0.29, 0.41, 1.71, 0.73,

		// IdNumSurroundingQueen / IdOpponentNumSurroundingQueen
		-3.15, 3.86,

		// IdNumCanMove
		0.75, 0.0, 0.57, 0.0, -0.07, 0.0, 1.14, 0.0, 0.07, 0.0,
		// IdOpponentNumCanMove
		0.05, 0.0, -0.17, 0.0, 0.13, 0.0, 0.26, 0.0, 0.02, 0.0,

		// IdNumThreateningMoves
		0., 0.,

		// F_NUM_TO_DRAW
		0.,

		// IdNumSingle,
		0., 0.,

		// Bias: *Must always be last*
		-0.79,
	).WithName("v0")

	PreTrainedV1 = NewWithWeights(
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// IdNumOffboard
		0.04304, 0.03418, 0.04503, -1.863, 0.0392,
		// IdOpponentNumOffboard
		0.05537, 0.03768, 0.04703, 1.868, 0.03902,

		// IdNumSurroundingQueen / IdOpponentNumSurroundingQueen
		-3.112, 3.422,

		// IdNumCanMove
		0.6302, 0.0, 0.4997, 0.0, -0.1359, 0.0, 1.115, 0.0, 0.0436, 0.0,

		// IdOpponentNumCanMove
		-0.001016, 0.0, -0.2178, 0.0, 0.05738, 0.0, 0.2827, 0.0, -0.01102, 0.0,

		// IdNumThreateningMoves
		-0.1299, -0.04499,

		// IdOpponentNumThreateningMoves
		// 0.1299, 0.04499,

		// F_NUM_TO_DRAW
		0.00944,

		// IdNumSingle,
		0., 0.,

		// Bias: *Must always be last*
		-0.8161,
	).WithName("v1")

	PreTrainedV2 = NewWithWeights(
		// NumOffboard -> 5
		0.0446, 0.0340, 0.0287, -1.8639, 0.0321,

		// OppNumOffboard -> 5
		0.0532, 0.0373, 0.0572, 1.8688, 0.0432,

		// IdNumSurroundingQueen -> 1
		-3.0338,

		// OppNumSurroundingQueen -> 1
		3.3681,

		// NumCanMove -> 10
		0.5989, 0.0090, 0.4845, -0.0045, -0.1100, 0.0213, 1.0952, -0.0198, 0.0468, 0.0054,

		// OppNumCanMove -> 10
		0.0185, -0.0096, -0.2016, 0.0043, 0.0483, -0.0133, 0.2939, 0.0113, 0.0004, 0.0037,

		// IdNumThreateningMoves -> 2
		-0.0946, 0.0114,

		// MovesToDraw -> 1
		0.0033,

		// IdNumSingle,
		0., 0.,

		// Bias -> 1
		-0.8147,
	).WithName("v2")

	PreTrainedV3 = NewWithWeights(
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// NumOffboard -> 5
		-0.1891, 0.0648, -0.0803, -1.8169, -0.0319,

		// OppNumOffboard -> 5
		0.2967, 0.0625, 0.2306, 2.2038, 0.1646,

		// IdNumSurroundingQueen -> 1
		-2.4521,

		// OppNumSurroundingQueen -> 1
		2.7604,

		// NumCanMove -> 10
		0.0065, 0.4391, 0.5519, -0.4833, -0.0156, 0.1361, 0.9591, -0.1592, 0.2405, 0.0343,

		// OppNumCanMove -> 10
		0.2158, -0.7518, -0.4865, 0.3333, 0.1677, -0.2764, -0.0134, -0.2725, 0.0145, 0.0184,

		// IdNumThreateningMoves -> 2
		0.0087, 0.0714,

		// OppNumThreateningMoves -> 2
		0.1979, 0.0486,

		// MovesToDraw -> 1
		-0.0074,

		// NumSingle -> 2
		-0.2442, 0.3755,

		// QueenIsCovered -> 2
		0, 0, // -10, 10,

		// AverageDistanceToQueen
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		0, -0.01, -0.01, 0, -0.01,

		// OppAverageDistanceToQueen
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		0, 0.001, 0.001, 0, 0.001,

		// Bias -> 1
		-0.7409,
	).WithName("v3")

	PreTrainedV4 = NewWithWeights(
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// NumOffboard -> 5
		-0.1891, 0.0648, -0.0803, -1.8169, -0.0319,

		// OppNumOffboard -> 5
		0.2967, 0.0625, 0.2306, 2.2038, 0.1646,

		// IdNumSurroundingQueen -> 1
		-2.4521,

		// OppNumSurroundingQueen -> 1
		2.7604,

		// NumCanMove -> 10
		0.0065, 0.4391, 0.5519, -0.4833, -0.0156, 0.1361, 0.9591, -0.1592, 0.2405, 0.0343,

		// OppNumCanMove -> 10
		0.2158, -0.7518, -0.4865, 0.3333, 0.1677, -0.2764, -0.0134, -0.2725, 0.0145, 0.0184,

		// IdNumThreateningMoves -> 2
		0.0087, 0.0714,

		// OppNumThreateningMoves -> 2
		0.1979, 0.0486,

		// MovesToDraw -> 1
		-0.0074,

		// NumSingle -> 2
		-0.2442, 0.3755,

		// QueenIsCovered -> 2
		0, 0, // -10, 10,

		// AverageDistanceToQueen
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		0, -0.01, -0.01, 0, -0.01,

		// OppAverageDistanceToQueen
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		0, 0.001, 0.001, 0, 0.001,

		// NumPlacementPositions,
		0.1, -0.1,

		// Bias -> 1
		-0.7409,
	).WithName("v4")

	// PreTrainedBest is an alias to the current best linear model.
	PreTrainedBest = PreTrainedV3.Clone().WithName("best")
)

// NewFromParams returns the linear scorer if "linear" is set, otherwise it returns nil (and no error).
// It returns an error if an unknown model or if it is a path to file, and it can't load or parse it.
func NewFromParams(params parameters.Params) (ai.BoardScorer, error) {
	if _, found := params["linear"]; !found {
		return nil, nil
	}
	modelName, err := parameters.PopParamOr(params, "linear", "best")
	if err != nil {
		return nil, err
	}
	var selected *Scorer
	for _, scorer := range []*Scorer{
		PreTrainedBest, PreTrainedV0, PreTrainedV1, PreTrainedV2, PreTrainedV3, PreTrainedV4} {
		if modelName == scorer.name {
			selected = scorer
		}
	}
	if selected == nil {
		selected, err = LoadOrCreate(modelName)
		if err != nil {
			err = errors.WithMessagef(err, "failed to load model \"linear=%s\"", modelName)
			return nil, err
		}
	}

	klog.V(1).Infof("Linear model %s with %d features\n", selected, selected.Version())
	// TODO: add support for hyper-parameters from params.
	return selected, nil
}
