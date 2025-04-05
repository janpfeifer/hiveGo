package gomlx

import (
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/hiveGo/internal/features"
	"github.com/janpfeifer/hiveGo/internal/state"
)

// CreatePolicyInputs implements PolicyModel.CreateInputs.
// It will input the feature values for the current board, and each of the next
// board positions.
func (fnn *FNN) CreatePolicyInputs(boards *state.Board) []*tensors.Tensor {
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

// CreatePolicyLabels implements PolicyModel.CreateLabels.
func (fnn *FNN) CreatePolicyLabels(scoreLabel float32, policyLabels []float32) *tensors.Tensor {
	paddedBatchSize := fnn.paddedBatchSize(len(labels))
	boardLabels := tensors.FromShape(shapes.Make(dtypes.Float32, paddedBatchSize, 1))
	tensors.MutableFlatData(boardLabels, func(flat []float32) {
		copy(flat, labels)
	})
	return boardLabels
}
