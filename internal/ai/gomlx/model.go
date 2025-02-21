package gomlx

import (
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/janpfeifer/hiveGo/internal/state"
)

// Model is a GoMLX supported model, the backend of the gomlx.Scorer.
type Model interface {
	// Context used by the model: with both it weights and hyperparameters.
	Context() *context.Context

	// CreateInputs for a batch of boards as tensors.
	CreateInputs(boards []*state.Board) []*tensors.Tensor
}
