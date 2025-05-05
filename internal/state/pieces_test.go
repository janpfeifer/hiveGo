package state_test

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	. "github.com/janpfeifer/hiveGo/internal/state/statetest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestQueenMoves(t *testing.T) {
	layout := []PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, SPIDER},
		{Pos{-1, 1}, 1, GRASSHOPPER},
		{Pos{2, 0}, 0, QUEEN},
		{Pos{-1, 2}, 1, GRASSHOPPER},
	}
	board := BuildBoard(layout, false)
	board.BuildDerived()
	PrintBoard(board)

	queenMoves := listMovesForPiece(board, QUEEN, Pos{2, 0})
	want := []Pos{{2, -1}, {1, 1}}
	require.Equal(t, want, queenMoves)

	// Now queen can't move because it would break hive.
	layout = append(layout, PieceOnBoard{Pos{3, 0}, 0, GRASSHOPPER})
	board = BuildBoard(layout, false)
	fmt.Println("\n\t> Test queen can't move because it would break hive")
	PrintBoard(board)
	board.BuildDerived()
	queenMoves = listMovesForPiece(board, QUEEN, Pos{2, 0})
	require.Empty(t, queenMoves)

	// If we put some pieces around it can move again.
	layout = append(layout, PieceOnBoard{Pos{1, 1}, 0, BEETLE})
	layout = append(layout, PieceOnBoard{Pos{2, 1}, 0, BEETLE})
	board = BuildBoard(layout, false)
	fmt.Println("\n\t> Test queen can move again after putting pieces around it")
	PrintBoard(board)
	board.BuildDerived()
	queenMoves = listMovesForPiece(board, QUEEN, Pos{2, 0})
	want = []Pos{{2, -1}, {3, -1}}
	require.Equal(t, want, queenMoves)

	// Finally piece is not supposed to squeeze among pieces:
	layout = append(layout, PieceOnBoard{Pos{2, -1}, 0, ANT})
	board = BuildBoard(layout, false)
	fmt.Println("\n\t> Test queen can't squeeze among pieces")
	PrintBoard(board)
	board.BuildDerived()
	queenMoves = listMovesForPieceDisplayPos(board, QUEEN, Pos{2, 1})
	require.Empty(t, queenMoves)
}

func TestSpiderMoves(t *testing.T) {
	layout := []PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, SPIDER},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 0}, 0, QUEEN},
		{Pos{-1, 2}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, SPIDER},
		{Pos{-1, 3}, 0, SPIDER},
	}
	board := BuildBoard(layout, false)
	PrintBoard(board)
	board.BuildDerived()

	// Spider at (1,0) should be locked in place (it would break the hive)
	spiderMoves := listMovesForPiece(board, SPIDER, Pos{1, 0})
	if len(spiderMoves) != 0 {
		t.Errorf("Wanted Spider moves to be empty, got %v", spiderMoves)
	}

	// Spider at (1,1) is free to move.
	want := []Pos{{3, -1}, {0, 3}}
	spiderMoves = listMovesForPiece(board, SPIDER, Pos{1, 1})
	assert.Equal(t, want, spiderMoves)

	// Spider at (-1, 3)
	want = []Pos{{-2, 1}, {2, 1}}
	// PrintBoard(board)
	spiderMoves = listMovesForPiece(board, SPIDER, Pos{-1, 3})
	assert.Equal(t, want, spiderMoves)
}

func TestGrasshopperMoves(t *testing.T) {
	layout := []PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, QUEEN},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 0}, 0, GRASSHOPPER},
		{Pos{-1, 2}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, GRASSHOPPER},
		{Pos{-1, 3}, 1, GRASSHOPPER},
	}
	board := BuildBoard(layout, false)
	PrintBoard(board)

	tests := [2][]generics.Pair[Pos, []Pos]{
		{ // Player0
			{Pos{1, 1}, []Pos{{1, -1}, {3, -1}}},
			{Pos{2, 0}, []Pos{{-2, 0}, {0, 2}}},
		},
		{ // Player1
			{Pos{-1, 2}, nil},
			{Pos{-1, 3}, []Pos{{-1, -1}}},
		},
	}
	for playerIdx := range PlayerNum(2) {
		board.NextPlayer = playerIdx
		board.BuildDerived()
		for _, test := range tests[playerIdx] {
			grasshopperMoves := listMovesForPiece(board, GRASSHOPPER, test.First)
			want := test.Second
			assert.Equalf(t, want, grasshopperMoves, "For player #%d grasshopper at %s, wanted %v, got %v",
				playerIdx, test.First, want, grasshopperMoves)
		}
	}
}

func TestAntMoves(t *testing.T) {
	layout := []PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 1}, 1, BEETLE},
		{Pos{1, 0}, 0, QUEEN},
		{Pos{-1, 2}, 1, QUEEN},
		{Pos{2, 0}, 0, ANT},
		{Pos{-1, 3}, 1, ANT},
		{Pos{1, 1}, 0, ANT},
		{Pos{-2, 4}, 1, ANT},
		{Pos{0, 1}, 0, ANT},
	}
	board := BuildBoard(layout, false)
	PrintBoard(board)

	tests := [2][]generics.Pair[Pos, []Pos]{
		{ // Player #0
			{Pos{1, 1}, []Pos{
				{0, -1}, {1, -1}, {2, -1}, {3, -1}, {-1, 0}, {3, 0},
				{-2, 1}, {2, 1}, {-2, 2}, {0, 2},
				{-2, 3}, {0, 3},
				{-3, 4}, {-1, 4}, {-3, 5}, {-2, 5}}},
			{Pos{2, 0}, []Pos{
				{0, -1}, {1, -1}, {2, -1}, {-1, 0},
				{-2, 1}, {2, 1},
				{-2, 2}, {0, 2}, {1, 2},
				{-2, 3}, {0, 3},
				{-3, 4}, {-1, 4}, {-3, 5}, {-2, 5}}},
			//{Pos{0, 1}, nil},
		},
		{ // Player #1
			{Pos{-1, 3}, nil}, // It would break the hive during the move.
			{Pos{-2, 4}, []Pos{
				{0, -1}, {1, -1}, {2, -1}, {3, -1}, {-1, 0}, {3, 0},
				{-2, 1}, {2, 1}, {-2, 2}, {0, 2}, {1, 2},
				{-2, 3}, {0, 3},
				{-1, 4}}},
		},
	}
	for playerIdx := range PlayerNum(2) {
		board.NextPlayer = playerIdx
		board.BuildDerived()
		for _, test := range tests[playerIdx] {
			antMoves := listMovesForPiece(board, ANT, test.First)
			want := test.Second
			SortPositions(want)
			assert.Equalf(t, want, antMoves, "For player #%d ant at %s, wanted %v, got %v",
				playerIdx, test.First, want, antMoves)
		}
	}
}

func TestBeetleMoves(t *testing.T) {
	layout := []PieceOnBoard{
		{Pos{0, 0}, 0, BEETLE},
		{Pos{0, -1}, 1, ANT},
		{Pos{0, 1}, 0, SPIDER},
		{Pos{1, -2}, 1, BEETLE},
		{Pos{1, 0}, 0, BEETLE},
		{Pos{1, -3}, 1, QUEEN},
		{Pos{0, 2}, 0, QUEEN},
		{Pos{2, -2}, 1, SPIDER},
		{Pos{2, -1}, 0, ANT},
	}
	board := BuildBoard(layout, false)
	PrintBoard(board)
	board.BuildDerived()
	pieceType := BEETLE

	tests := [2][]generics.Pair[Pos, []Pos]{
		{ // Player #0
			{Pos{0, 0}, []Pos{{0, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}}},
			{Pos{1, 0}, []Pos{{2, -1}, {0, 0}, {2, 0}, {0, 1}, {1, 1}}},
		},
		{ // Player #1
			{Pos{1, -2}, nil}, // No moves
		},
	}

	for playerIdx := range PlayerNum(2) {
		board.NextPlayer = playerIdx
		board.BuildDerived()
		for _, test := range tests[playerIdx] {
			moves := listMovesForPiece(board, pieceType, test.First)
			want := test.Second
			SortPositions(want)
			assert.Equalf(t, want, moves, "For player #%d %s at %s, wanted %v, got %v",
				playerIdx, pieceType, test.First, want, moves)
		}
	}

	// Checks that the Beetle cannot make a move "in the air" (to Pos{-2, 1}).
	fmt.Println()
	layout = []PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{1, 0}, 0, QUEEN},
		{Pos{-3, 0}, 0, SPIDER},
		{Pos{-1, 1}, 0, BEETLE},

		{Pos{0, -1}, 1, ANT},
		{Pos{-1, -1}, 1, ANT},
		{Pos{0, -2}, 1, ANT},
		{Pos{-2, 0}, 1, QUEEN},
	}
	board = BuildBoard(layout, false)
	PrintBoard(board)
	board.NextPlayer = 0
	board.BuildDerived()
	moves := listMovesForPiece(board, BEETLE, Pos{-1, 1})
	want := []Pos{{-1, 0}, {0, 0}, {0, 1}}
	assert.Equalf(t, want, moves, "For player #%d %s at %s, wanted %v, got %v",
		0, pieceType, Pos{-1, 1}, want, moves)

	// Checks that the Beetle can move anywhere in the neighborhood if it is on top of another piece.
	layout = []PieceOnBoard{
		{Pos{0, 0}, 0, QUEEN},
		{Pos{0, 1}, 1, GRASSHOPPER},
		{Pos{0, 1}, 0, BEETLE},
		{Pos{0, 2}, 1, QUEEN},
	}
	board = BuildBoard(layout, false)
	PrintBoard(board)
	board.NextPlayer = 0
	board.BuildDerived()
	moves = listMovesForPiece(board, BEETLE, Pos{0, 1})
	want = []Pos{Pos{0, 0}, Pos{1, 0}, Pos{-1, 1}, Pos{1, 1}, Pos{-1, 2}, Pos{0, 2}}
	assert.Equalf(t, want, moves, "For player #%d %s at %s, wanted %v, got %v",
		0, pieceType, Pos{-1, 1}, want, moves)
}
