package gomlx

import (
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	fnnLayer "github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/layers/regularizer"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"
	"github.com/gomlx/compute/dtypes"
	"github.com/janpfeifer/hiveGo/internal/features"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
)

// AlphaZeroFNN implement a feed-forward model for scoring a board and its actions.
// It implements a PolicyModel.
type AlphaZeroFNN struct {
	scope *model.Scope
}

// Compile-time assert that AlphaZeroFNN implements PolicyModel.
var _ PolicyModel = &AlphaZeroFNN{}

// NewAlphaZeroFNN creates an AlphaZeroFNN model with a fresh context, initialized with hyperparameters set to their defaults.
func NewAlphaZeroFNN() *AlphaZeroFNN {
	store := model.NewStore()
	scope := store.RootScope()
	_ = store.ResetRNGState()
	scope.SetParams(map[string]any{
		"batch_size": 128,

		// Number of board features to extract.
		// This allows backward compatibility, otherwise better leave as is.
		"features_version": features.BoardFeaturesDim,

		optimizer.ParamOptimizer:       "adam",
		optimizer.ParamLearningRate:    0.001,
		optimizer.ParamAdamEpsilon:     1e-7,
		optimizer.ParamAdamDType:       "",
		cosineschedule.ParamPeriodSteps: 0,
		activation.ParamActivation:     "sigmoid",
		layers.ParamDropoutRate:         0.0,
		regularizer.ParamL2:            1e-5,
		regularizer.ParamL1:            1e-5,

		// AlphaZeroFNN network parameters:
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
	return &AlphaZeroFNN{scope: scope}
}

// Clone model. Notice the associated checkpoint.Handler is not copied, and have to be set again.
func (fnn *AlphaZeroFNN) Clone() PolicyModel {
	clonedStore, err := fnn.scope.Store().Clone()
	if err != nil {
		panic(errors.Wrap(err, "failed to clone AlphaZeroFNN store"))
	}
	return &AlphaZeroFNN{scope: clonedStore.RootScope()}
}

func (fnn *AlphaZeroFNN) Context() *model.Scope {
	return fnn.scope
}

// paddedSize returns a padded batchSize for the given numBoards.
// This is important so we don't have too many different versions of the program for every different batch size.
func (fnn *AlphaZeroFNN) paddedSize(numBoards int) int {
	if numBoards == 1 {
		// Always have the option to support 1.
		return numBoards
	}
	// Make sure the default batchSize is supported without padding.
	defaultBatchSize := model.GetParamOr(fnn.scope, "batch_size", 128)
	if numBoards == defaultBatchSize {
		return numBoards
	}

	// Starts with 8, anything smaller than that, the cost in space is too small, not worth having multiple programs
	// for different padding sizes.
	paddedSize := 8
	for paddedSize < numBoards {
		// Increase 1.5x at a time.
		paddedSize = paddedSize + (paddedSize+1)/2
	}
	return paddedSize
}

// CreateValueInputs implements PolicyModel.CreateValueInputs.
func (fnn *AlphaZeroFNN) CreateValueInputs(board *state.Board) []*tensors.Tensor {
	boardsFeatures := fnn.createBoardsFeatures([]*state.Board{board}, 0)
	numBoards := tensors.FromScalar(int32(1))
	return []*tensors.Tensor{boardsFeatures, numBoards}
}

// Create raw features for a set of boards (not its actions), maybe with padding.
// minPadding is the minimal amount of padding to make sure is included.
func (fnn *AlphaZeroFNN) createBoardsFeatures(boards []*state.Board, minPadding int) *tensors.Tensor {
	version := model.GetParamOr(fnn.scope, "features_version", features.BoardFeaturesDim)
	numBoards := len(boards)
	paddedBatchSize := fnn.paddedSize(numBoards + minPadding)
	boardFeatures := tensors.FromShape(shapes.Make(dtypes.Float32, paddedBatchSize, version))
	tensors.MutableFlatData(boardFeatures, func(flat []float32) {
		for boardIdx, board := range boards {
			values := features.ForBoard(board, version)
			copy(flat[boardIdx*version:], values)
		}
	})
	return boardFeatures
}

// CreatePolicyInputs implements PolicyModel.
func (fnn *AlphaZeroFNN) CreatePolicyInputs(boards []*state.Board) []*tensors.Tensor {
	// Board features:
	boardFeatures := fnn.createBoardsFeatures(boards, 1)
	numBoardsT := tensors.FromScalar(int32(len(boards)))

	// Action features: calculated from the boards after playing each action.
	var numActions int
	for _, board := range boards {
		numActions += board.NumActions()
	}
	numActionsT := tensors.FromScalar(int32(numActions))
	// Collect all actions boards.
	actionsBoards := make([]*state.Board, 0, numActions)
	for _, board := range boards {
		actionsBoards = append(actionsBoards, board.TakeAllActions()...)
	}
	actionsFeatures := fnn.createBoardsFeatures(actionsBoards, 0)
	// Create edges: a mapping from actionsIdx to boardIdx.
	numPaddedActions := actionsFeatures.Shape().Dim(0)
	actionsToBoardIdx := tensors.FromShape(shapes.Make(dtypes.Int32, numPaddedActions))
	tensors.MutableFlatData(actionsToBoardIdx, func(flat []int32) {
		actionIdx := 0
		for boardIdx, board := range boards {
			for _ = range board.NumActions() {
				flat[actionIdx] = int32(boardIdx)
				actionIdx++
			}
		}
		// Fill the padded actions indices to the dummy board index.
		dummyBoardIdx := int32(len(boards))
		for ; actionIdx < numPaddedActions; actionIdx++ {
			flat[actionIdx] = dummyBoardIdx
		}
	})
	return []*tensors.Tensor{boardFeatures, numBoardsT, actionsFeatures, actionsToBoardIdx, numActionsT}
}

func (fnn *AlphaZeroFNN) ForwardValueGraph(scope *model.Scope, valueInputs []*Node) (values *Node) {
	boardsFeatures, numBoards := valueInputs[0], valueInputs[1]
	boardEmbed := fnn.boardEmbedding(scope, boardsFeatures, numBoards)
	return fnn.boardValues(scope, boardEmbed)
}

func (fnn *AlphaZeroFNN) ForwardPolicyGraph(scope *model.Scope, policyInputs []*Node) (values *Node, policy *Node) {
	boardFeatures, numBoards, actionsFeatures, actionsToBoardIdx, numActions := policyInputs[0], policyInputs[1], policyInputs[2], policyInputs[3], policyInputs[4]
	numPaddedBoards := boardFeatures.Shape().Dim(0)
	numPaddedActions := actionsFeatures.Shape().Dim(0)

	// Base board tower is shared between value and actions (policy) logits.
	boardEmbed := fnn.boardEmbedding(scope, boardFeatures, numBoards)
	actionsEmbed := fnn.boardEmbedding(scope, actionsFeatures, numActions)

	// One-layer from board embeddings to its values (scores).
	values = fnn.boardValues(scope, boardEmbed)

	// Send message (like a MPNN, a type of graph neural network) from boards to actions.
	actionsBoardEmbed := Gather(boardEmbed, ExpandAxes(actionsToBoardIdx, -1))
	actionsBoardEmbed.AssertDims(numPaddedActions, boardEmbed.Shape().Dim(1))
	actionsEmbed = Concatenate([]*Node{actionsEmbed, actionsBoardEmbed}, -1)

	// actions (policy) tower
	actionsCtx := scope.At("actions")
	var actionsLogits *Node
	if model.GetParamOr(scope, "kan", false) {
		// Use KAN, all configured by context hyperparameters. See createDefaultContext for defaults.
		actionsLogits = kan.New(actionsCtx.At("kan"), actionsEmbed, 1).Done()
	} else {
		// Normal AlphaZeroFNN, all configured by context hyperparameters. See createDefaultContext for defaults.
		actionsLogits = fnnLayer.New(actionsCtx.At("fnn"), actionsEmbed, 1).Done()
	}
	actionsLogits = Reshape(actionsLogits, -1) // Flatten
	policyRagged := MakeRagged2D(numPaddedBoards, actionsLogits, actionsToBoardIdx).Softmax()
	policy = policyRagged.Flat
	return
}

func (fnn *AlphaZeroFNN) boardEmbedding(scope *model.Scope, boardFeatures, numBoards *Node) *Node {
	scope = scope.At("base_board_tower")
	embeddings := boardFeatures
	boardEmbedDim := model.GetParamOr(scope, fnnLayer.ParamNumHiddenNodes, 4)
	// ValueModel itself is an AlphaZeroFNN or a KAN.
	if model.GetParamOr(scope, "kan", false) {
		// Use KAN, all configured by context hyperparameters. See createDefaultContext for defaults.
		embeddings = kan.New(scope.At("kan"), embeddings, boardEmbedDim).Done()
	} else {
		// Normal AlphaZeroFNN, all configured by context hyperparameters. See createDefaultContext for defaults.
		embeddings = fnnLayer.New(scope.At("fnn"), embeddings, boardEmbedDim).Done()
	}

	// Zero masked out elements.
	batchSize := embeddings.Shape().Dim(0)
	if batchSize == 1 {
		// No padding
		return embeddings
	}
	mask := fnn.getMask(embeddings, numBoards)
	embeddings = Where(Squeeze(mask, -1), embeddings, ZerosLike(embeddings))
	return embeddings
}

func (fnn *AlphaZeroFNN) boardValues(scope *model.Scope, boardEmbed *Node) *Node {
	scope = scope.At("board_output")
	var logits *Node
	if model.GetParamOr(scope, "kan", false) {
		// Use KAN, all configured by context hyperparameters. See createDefaultContext for defaults.
		logits = kan.New(scope.At("kan"), boardEmbed, 1).
			NumHiddenLayers(0, 0).Done()
	} else {
		// Normal AlphaZeroFNN, all configured by context hyperparameters. See createDefaultContext for defaults.
		logits = fnnLayer.New(scope.At("kan"), boardEmbed, 1).
			NumHiddenLayers(0, 0).Done()
	}
	return Tanh(logits)
}

// CreatePolicyLabels implements PolicyModel.
func (fnn *AlphaZeroFNN) CreatePolicyLabels(boardLabels []float32, policyLabels [][]float32) []*tensors.Tensor {
	paddedBatchSize := fnn.paddedSize(len(boardLabels) + 1)
	boardLabelsT := tensors.FromShape(shapes.Make(dtypes.Float32, paddedBatchSize))
	tensors.MutableFlatData(boardLabelsT, func(flat []float32) {
		copy(flat, boardLabels)
	})

	var numActions int
	for _, boardPolicy := range policyLabels {
		numActions += len(boardPolicy)
	}
	paddedNumActions := fnn.paddedSize(numActions)
	policyLabelsT := tensors.FromShape(shapes.Make(dtypes.Float32, paddedNumActions))
	tensors.MutableFlatData(policyLabelsT, func(flat []float32) {
		actionIdx := 0
		for _, boardPolicy := range policyLabels {
			for _, label := range boardPolicy {
				flat[actionIdx] = label
				actionIdx++
			}
		}
	})
	return []*tensors.Tensor{boardLabelsT, policyLabelsT}
}

func (fnn *AlphaZeroFNN) LossGraph(scope *model.Scope, inputs []*Node, labels []*Node) *Node {
	numBoards, numActions := inputs[1], inputs[4]
	predictedValues, predictedPolicies := fnn.ForwardPolicyGraph(scope, inputs)
	boardLabels, policyLabels := labels[0], labels[1]
	if boardLabels.Rank() == 1 {
		boardLabels = ExpandAxes(boardLabels, -1)
	}

	// Board labels part:
	boardsMask := Squeeze(fnn.getMask(predictedValues, numBoards), -1)
	boardLosses := ReduceAllMean(loss.MeanSquaredError([]*Node{boardLabels, boardsMask}, []*Node{predictedValues}))

	// Policy losses: we do the cross-entropy loss manually, to handle the raggedness in the mean.
	// We want each board to have the same weight on the final loss.
	policiesMask := Squeeze(fnn.getMask(policyLabels, numActions), -1)
	predictedPolicies = ExpandAxes(predictedPolicies, -1)
	policyLabels = ExpandAxes(policyLabels, -1)
	policyLosses := loss.CategoricalCrossEntropy([]*Node{policyLabels, policiesMask},
		[]*Node{predictedPolicies})
	return Add(boardLosses, policyLosses)
}

// getMask of a batch, given the number of used elements (numUsed, an Int32 scalar) from it.
func (fnn *AlphaZeroFNN) getMask(batch, numUsed *Node) *Node {
	g := batch.Graph()
	batchSize := batch.Shape().Dim(0)
	batchMask := LessThan(Iota(g, shapes.Make(dtypes.Int32, batchSize, 1), 0), numUsed)
	return batchMask
}
