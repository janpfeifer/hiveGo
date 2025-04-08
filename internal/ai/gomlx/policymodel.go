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

	// CreateValueInputs when evaluating the board value. No batch supported here.
	CreateValueInputs(board *state.Board) []*tensors.Tensor

	// CreatePolicyInputs when evaluating the board's policy: the actions probabilities.
	// It should include any padding needed by the model.
	CreatePolicyInputs(board *state.Board) []*tensors.Tensor

	// CreatePolicyLabels tensors used during training.
	CreatePolicyLabels(scoreLabel float32, policyLabels []float32) []*tensors.Tensor

	// ForwardValueGraph outputs only the value score of a board.
	ForwardValueGraph(ctx *context.Context, valueInputs []*graph.Node) (value *graph.Node)

	// ForwardPolicyGraph is the GoMLX model graph function with the forward path that includes
	// the value score of a board and its policy values (action probabilities).
	//
	// The returned policy probabilities may be padded -- just discard the values beyond the number of actions
	// for the board.
	ForwardPolicyGraph(ctx *context.Context, policyInputs []*graph.Node) (value *graph.Node, policy *graph.Node)

	// LossGraph should calculate the lossExec given the board inputs and the labels (shaped [batch_size, 1]).
	// It must return a scalar with the lossExec value.
	LossGraph(ctx *context.Context, inputs []*graph.Node, labels *graph.Node) *[]graph.Node
}
