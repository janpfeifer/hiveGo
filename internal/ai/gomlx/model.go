package gomlx

import (
	"github.com/gomlx/gomlx/graph"
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

	// ForwardGraph is the GoMLX model graph function with the forward path.
	// It must return the scores for each board, shaped [board].
	ForwardGraph(ctx *context.Context, inputs []*graph.Node) *graph.Node

	// LossGraph should calculate the lossExec given the board inputs and the labels (shaped [batch_size]).
	// It must return a scalar with the lossExec value.
	LossGraph(ctx *context.Context, inputs []*graph.Node, labels *graph.Node) *graph.Node
}
