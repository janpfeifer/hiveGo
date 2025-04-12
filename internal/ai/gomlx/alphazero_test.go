package gomlx

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/stretchr/testify/require"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

// PieceOnBoard represents a position and ownership of a piece in the board.
type PieceOnBoard struct {
	pos    Pos
	player PlayerNum
	piece  PieceType
}

// buildBoardFromLayout from a collection of pieces. Their positions may be in "display coordinates".
func buildBoardFromLayout(layout []PieceOnBoard, displayPos bool) (b *Board) {
	b = NewBoard()
	for _, p := range layout {
		pos := p.pos
		if displayPos {
			pos = pos.FromDisplayPos()
		}
		b.StackPiece(pos, p.player, p.piece)
		b.SetAvailable(p.player, p.piece, b.Available(p.player, p.piece)-1)
	}
	return
}

// buildBoardsForAlphaZeroFNN
func buildBoardsForAlphaZeroFNN(t *testing.T) (boards []*Board, numActions int) {
	board0 := NewBoard()
	board0.BuildDerived()
	require.Equal(t, 5, board0.NumActions()) // 5 possible pieces to put as the first action.
	board1 := buildBoardFromLayout([]PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, SPIDER},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 0}, 0, QUEEN},
		{Pos{-1, 2}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, SPIDER},
		{Pos{-1, 3}, 1, SPIDER},
	}, false)
	board1.BuildDerived()
	require.Equal(t, 22, board1.NumActions())
	boards = []*Board{board0, board0, board1} // 3 boards, the board0 appears twice.
	numActions = 2*board0.NumActions() + board1.NumActions()
	return
}

func TestAlphaZeroFNN_Padding(t *testing.T) {
	fnn := NewAlphaZeroFNN()
	wantPaddedSizes := []int{1, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 18, 18, 18, 18, 18, 18, 27, 27, 27, 27, 27, 27, 27, 27, 27, 41, 41, 41, 41}
	gotPaddedSizes := make([]int, len(wantPaddedSizes))
	for ii := range wantPaddedSizes {
		gotPaddedSizes[ii] = fnn.paddedSize(ii + 1)
	}
	fmt.Printf("Padded sizes: %#v\n", gotPaddedSizes)
	require.Equal(t, wantPaddedSizes, gotPaddedSizes)
}

func TestAlphaZeroFNN_Inputs(t *testing.T) {
	boards, numActions := buildBoardsForAlphaZeroFNN(t)
	fnn := NewAlphaZeroFNN()
	inputs := fnn.CreatePolicyInputs(boards)

	require.Len(t, inputs, 5)
	boardFeaturesT, numBoardsT, actionFeaturesT, actionsToBoardIdxT, numActionsT := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	require.Equal(t, int32(len(boards)), tensors.ToScalar[int32](numBoardsT))
	paddedNumBoards := fnn.paddedSize(len(boards) + 1)
	require.Equal(t, paddedNumBoards, boardFeaturesT.Shape().Dim(0))
	require.Equal(t, int32(numActions), tensors.ToScalar[int32](numActionsT))

	paddedNumActions := fnn.paddedSize(numActions)
	require.Equal(t, 2, actionFeaturesT.Rank())
	require.Equal(t, paddedNumActions, actionFeaturesT.Shape().Dim(0))
	require.Equal(t, boardFeaturesT.Shape().Dim(1), actionFeaturesT.Shape().Dim(1))

	// Indices from actions back to the board number.
	// Notice that the padded space at the end points to the dummy board #3 (one past end).
	actionsToBoardIdx := []int32{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3}
	require.Equal(t, actionsToBoardIdx, tensors.CopyFlatData[int32](actionsToBoardIdxT))
}

func TestAlphaZeroFNN_ForwardPolicyGraph(t *testing.T) {
	boards, numActions := buildBoardsForAlphaZeroFNN(t)
	fnn := NewAlphaZeroFNN()
	numPaddedBoards := fnn.paddedSize(len(boards) + 1)
	numPaddedActions := fnn.paddedSize(numActions)
	inputs := fnn.CreatePolicyInputs(boards)
	inputsAny := generics.SliceMap(inputs, func(t *tensors.Tensor) any { return t })
	backend := graphtest.BuildTestBackend()
	outputs := context.ExecOnceN(backend, fnn.Context(), func(ctx *context.Context, inputs []*graph.Node) []*graph.Node {
		values, policies := fnn.ForwardPolicyGraph(ctx, inputs)
		return []*graph.Node{values, policies}
	}, inputsAny...)
	valuesT, policiesT := outputs[0], outputs[1]
	fmt.Printf("Values: %s\n", valuesT)
	fmt.Printf("Policies: %s\n", policiesT)

	valuesT.Shape().AssertDims(numPaddedBoards, 1)
	policiesT.Shape().AssertDims(numPaddedActions)
	policies := tensors.CopyFlatData[float32](policiesT)

	// Makes sure policies sum to 1
	var actionIdx int
	for _, board := range boards {
		var sumProbs float32
		for _ = range board.NumActions() {
			sumProbs += policies[actionIdx]
			actionIdx++
		}
		require.InDeltaf(t, 1.0, sumProbs, 1e-4, "Sum of probabilities for board %s is %.3f, it should be 1", board, sumProbs)
	}
}

func TestAlphaZeroFNN_LossGraph(t *testing.T) {
	boards, _ := buildBoardsForAlphaZeroFNN(t)
	valuesLabels := []float32{0, 0.8, -0.8}
	policyLabels := [][]float32{
		{0.2, 0.2, 0.2, 0.2, 0.2},
		{1, 0, 0, 0, 0},
		{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5},
	}
	fnn := NewAlphaZeroFNN()
	policyInputs := fnn.CreatePolicyInputs(boards)
	labelsInputs := fnn.CreatePolicyLabels(valuesLabels, policyLabels)
	inputsAny := generics.SliceMap(append(policyInputs, labelsInputs...), func(t *tensors.Tensor) any { return t })
	backend := graphtest.BuildTestBackend()
	lossT := context.ExecOnce(backend, fnn.Context(), func(ctx *context.Context, inputs []*graph.Node) *graph.Node {
		policyInputsN := inputs[:len(policyInputs)]
		labelsInputsN := inputs[len(policyInputs):]
		return fnn.LossGraph(ctx, policyInputsN, labelsInputsN)
	}, inputsAny...)
	fmt.Printf("Loss: %s\n", lossT)
	lossT.Shape().AssertScalar()
}
