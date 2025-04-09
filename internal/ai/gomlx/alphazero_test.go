package gomlx

import (
	"fmt"
	"github.com/gomlx/gomlx/types/tensors"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/stretchr/testify/require"
	"testing"
)

// PieceOnBoard represents a position and ownership of a piece in the board.
type PieceOnBoard struct {
	pos    Pos
	player PlayerNum
	piece  PieceType
}

// buildBoard from a collection of pieces. Their positions may be in "display coordinates".
func buildBoard(layout []PieceOnBoard, displayPos bool) (b *Board) {
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

func TestPadding(t *testing.T) {
	fnn := NewAlphaZeroFNN()
	wantPaddedSizes := []int{1, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 18, 18, 18, 18, 18, 18, 27, 27, 27, 27, 27, 27, 27, 27, 27, 41, 41, 41, 41}
	gotPaddedSizes := make([]int, len(wantPaddedSizes))
	for ii := range wantPaddedSizes {
		gotPaddedSizes[ii] = fnn.paddedSize(ii + 1)
	}
	fmt.Printf("Padded sizes: %#v\n", gotPaddedSizes)
	require.Equal(t, wantPaddedSizes, gotPaddedSizes)
}

func TestInputs(t *testing.T) {
	board0 := NewBoard()
	board0.BuildDerived()
	require.Equal(t, 5, board0.NumActions()) // 5 possible pieces to put as the first action.
	board1 := buildBoard([]PieceOnBoard{
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
	boards := []*Board{board0, board0, board1} // 3 boards, the board0 appears twice.

	fnn := NewAlphaZeroFNN()
	inputs := fnn.CreatePolicyInputs(boards)

	require.Len(t, inputs, 5)
	boardFeaturesT, numBoardsT, actionFeaturesT, actionsToBoardIdxT, numActionsT := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	require.Equal(t, int32(len(boards)), tensors.ToScalar[int32](numBoardsT))
	paddedNumBoards := fnn.paddedSize(len(boards) + 1)
	require.Equal(t, paddedNumBoards, boardFeaturesT.Shape().Dim(0))

	numActions := 2*board0.NumActions() + board1.NumActions()
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
