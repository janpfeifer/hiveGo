package gomlx

import (
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/janpfeifer/hiveGo/internal/state"
)

// PolicyModel is a GoMLX supported policy model (use with MCTS/AlphaZero), which is able to estimate both
// the value of a board position, and its policy: the probability of each action.
type PolicyModel interface {
	// Context used by the model: with both it weights and hyperparameters.
	Context() *context.Context

	// Clone returns a copy of the model with a cloned Context.
	Clone() PolicyModel

	// CreateValueInputs when evaluating the board value. No batch supported here.
	CreateValueInputs(board *state.Board) []*tensors.Tensor

	// CreatePolicyInputs when evaluating the board's policy: the actions probabilities.
	// It should include any padding needed by the model.
	//
	// It has to work with batch size of 1 board only (during playing the game), and with
	// arbitrary batch sizes (during training).
	//
	// The number of tensors returned is arbitrary for the model, but they must remain always fixed.
	CreatePolicyInputs(boards []*state.Board) []*tensors.Tensor

	// CreatePolicyLabels tensors used during training.
	// Labels are for a batch of boards.
	//
	// The number of tensors returned is arbitrary for the model, but they must remain always fixed.
	CreatePolicyLabels(scoreLabels []float32, policyLabels [][]float32) []*tensors.Tensor

	// ForwardValueGraph outputs only the value score of a board.
	ForwardValueGraph(ctx *context.Context, valueInputs []*graph.Node) (value *graph.Node)

	// ForwardPolicyGraph is the GoMLX model graph function with the forward path that includes
	// the value score of a board and its policy values (action probabilities).
	//
	// The returned policy probabilities is returned in a "ragged" format: the flat values of the actions probabilities
	// of all boards densely packed, with padding only in the end -- just discard the values beyond the total number of
	// actions for all the boards.
	ForwardPolicyGraph(ctx *context.Context, policyInputs []*graph.Node) (value *graph.Node, policy *graph.Node)

	// LossGraph should calculate the lossExec given the board inputs and the labels (shaped [batch_size, 1]).
	// It must return a scalar with the lossExec value -- if not a scalar, it is reduced with the mean.
	LossGraph(ctx *context.Context, inputs []*graph.Node, labels []*graph.Node) *graph.Node
}
