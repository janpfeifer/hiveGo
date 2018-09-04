package state_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = fmt.Printf

type PieceLayout struct {
	pos    Pos
	player uint8
	piece  Piece
}

func buildBoard(layout []PieceLayout) (b *Board) {
	b = NewBoard()
	for _, p := range layout {
		b.StackPiece(p.pos, p.player, p.piece)
		b.SetAvailable(p.player, p.piece, b.Available(p.player, p.piece)-1)
	}
	return
}

func listMovesForPiece(b *Board, piece Piece, pos Pos) (poss []Pos) {
	poss = nil
	d := b.Derived
	for _, a := range d.Actions {
		if a.Move && a.SourcePos == pos && a.Piece == piece {
			poss = append(poss, a.TargetPos)
		}
	}
	PosSort(poss)
	return
}

func printBoard(b *Board) {
	ui := ascii_ui.NewUI(true, false)
	ui.PrintBoard(b)
}

func TestOccupiedNeighbours(t *testing.T) {
	board := buildBoard([]PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, SPIDER},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 1}, 0, QUEEN},
		{Pos{-1, 2}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, SPIDER},
		{Pos{-1, 3}, 1, SPIDER},
	})
	board.BuildDerived()

	want := map[Pos]bool{Pos{-1, 3}: true, Pos{1, 1}: true, Pos{2, 1}: true}
	if !reflect.DeepEqual(want, board.Derived.RemovablePieces) {
		t.Errorf("Wanted removable positions in %v, got %v", want, board.Derived.RemovablePieces)
	}
}

func TestQueenMoves(t *testing.T) {
	layout := []PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, SPIDER},
		{Pos{-1, 1}, 1, GRASSHOPPER},
		{Pos{2, 1}, 0, QUEEN},
		{Pos{-1, 2}, 1, GRASSHOPPER},
	}
	board := buildBoard(layout)
	board.BuildDerived()

	queenMoves := listMovesForPiece(board, QUEEN, Pos{2, 1})
	want := []Pos{{2, 0}, {1, 1}}
	if !reflect.DeepEqual(want, queenMoves) {
		t.Errorf("Wanted Queen moves to be %v, got %v", want, queenMoves)
	}

	// Now queen can't move because it would break hive.
	layout = append(layout, PieceLayout{Pos{3, 1}, 0, GRASSHOPPER})
	board = buildBoard(layout)
	board.BuildDerived()
	queenMoves = listMovesForPiece(board, QUEEN, Pos{2, 1})
	if len(queenMoves) != 0 {
		t.Errorf("Wanted Queen moves to be empty, got %v", queenMoves)
	}

	// If we put some pieces around it can move again.
	layout = append(layout, PieceLayout{Pos{1, 1}, 0, BEETLE})
	layout = append(layout, PieceLayout{Pos{2, 2}, 0, BEETLE})
	board = buildBoard(layout)
	board.BuildDerived()
	queenMoves = listMovesForPiece(board, QUEEN, Pos{2, 1})
	want = []Pos{{2, 0}, {3, 0}}
	if !reflect.DeepEqual(want, queenMoves) {
		t.Errorf("Wanted Queen moves to be %v, got %v", want, queenMoves)
	}

	// Finally piece is not supposed to squeeze among pieces:
	layout = append(layout, PieceLayout{Pos{2, 0}, 0, ANT})
	board = buildBoard(layout)
	board.BuildDerived()
	queenMoves = listMovesForPiece(board, QUEEN, Pos{2, 1})
	if len(queenMoves) != 0 {
		t.Errorf("Wanted Queen moves to be empty, got %v", queenMoves)
	}
}

func TestSpiderMoves(t *testing.T) {
	layout := []PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, SPIDER},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 1}, 0, QUEEN},
		{Pos{-1, 2}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, SPIDER},
		{Pos{-1, 3}, 0, SPIDER},
	}
	board := buildBoard(layout)
	board.BuildDerived()
	// printBoard(board)

	// Spider at (1,0) should be locked in place (it would break the hive)
	spiderMoves := listMovesForPiece(board, SPIDER, Pos{1, 0})
	if len(spiderMoves) != 0 {
		t.Errorf("Wanted Spider moves to be empty, got %v", spiderMoves)
	}

	// Spider at (1,1) is free to move.
	want := []Pos{{3, 0}, {0, 3}, {0, 4}}
	spiderMoves = listMovesForPiece(board, SPIDER, Pos{1, 1})
	if !reflect.DeepEqual(want, spiderMoves) {
		t.Errorf("Wanted Spider moves to be %v, got %v", want, spiderMoves)
	}

	// Spider at (-1, 3)
	want = []Pos{{-2, 1}, {0, 2}, {1, 2}, {2, 2}}
	spiderMoves = listMovesForPiece(board, SPIDER, Pos{-1, 3})
	if !reflect.DeepEqual(want, spiderMoves) {
		t.Errorf("Wanted Spider moves to be %v, got %v", want, spiderMoves)
	}
}

func TestGrasshopperMoves(t *testing.T) {
	layout := []PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, QUEEN},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 1}, 0, GRASSHOPPER},
		{Pos{-1, 2}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, GRASSHOPPER},
		{Pos{-1, 3}, 1, GRASSHOPPER},
	}
	board := buildBoard(layout)
	board.BuildDerived()
	// printBoard(board)

	// Grasshoppers at (2,1) and (1,1) can jump to 2 directions.
	want := []Pos{{1, -1}, {3, 0}}
	grasshopperMoves := listMovesForPiece(board, GRASSHOPPER, Pos{1, 1})
	if !reflect.DeepEqual(want, grasshopperMoves) {
		t.Errorf("Wanted grasshopper moves to be %v, got %v", want, grasshopperMoves)
	}
	want = []Pos{{-1, -1}, {0, 2}}
	grasshopperMoves = listMovesForPiece(board, GRASSHOPPER, Pos{2, 1})
	if !reflect.DeepEqual(want, grasshopperMoves) {
		t.Errorf("Wanted grasshopper moves to be %v, got %v", want, grasshopperMoves)
	}

	grasshopperMoves = listMovesForPiece(board, GRASSHOPPER, Pos{-1, 2})
	if len(grasshopperMoves) != 0 {
		t.Errorf("Wanted grasshopper moves to be empty, got %v", grasshopperMoves)
	}
}

func TestAntMoves(t *testing.T) {
	layout := []PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, QUEEN},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 1}, 0, ANT},
		{Pos{-1, 2}, 1, ANT},
		{Pos{1, 1}, 0, ANT},
		{Pos{-1, 3}, 1, ANT},
	}
	board := buildBoard(layout)
	board.BuildDerived()
	// printBoard(board)

	// Ant at (1,1) can move anywhere connected to the hive.
	want := []Pos{{-2, 0}, {-2, 1}, {-2, 2}, {-2, 3}, {-2, 4}, {-1, -1}, {-1, 4},
		{0, -1}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, -1}, {2, 0}, {2, 2}, {3, 0}, {3, 1}}
	PosSort(want)
	antMoves := listMovesForPiece(board, ANT, Pos{1, 1})
	if !reflect.DeepEqual(want, antMoves) {
		t.Errorf("Wanted Ant moves to be %v, got %v", want, antMoves)
	}

	// Ant at (2,1) is can't squeeze between pieces into (0,1), but anywhere
	// connected to the hive should be fine.
	want = []Pos{{-2, 0}, {-2, 1}, {-2, 2}, {-2, 3}, {-2, 4}, {-1, -1}, {-1, 4},
		{0, -1}, {0, 2}, {0, 3}, {0, 4}, {1, -1}, {1, 2}, {2, 0}, {2, 2}}
	PosSort(want)
	antMoves = listMovesForPiece(board, ANT, Pos{2, 1})
	if !reflect.DeepEqual(want, antMoves) {
		t.Errorf("Wanted Ant moves to be %v, got %v", want, antMoves)
	}

	// Ant at (-1,2) should be blocked.
	antMoves = listMovesForPiece(board, ANT, Pos{-1, 2})
	if len(antMoves) != 0 {
		t.Errorf("Wanted Ant moves to be empty, got %v", antMoves)
	}
}

func TestBeetleMoves(t *testing.T) {
	layout := []PieceLayout{
		{Pos{0, 0}, 0, BEETLE},
		{Pos{0, -1}, 1, ANT},
		{Pos{0, 1}, 0, SPIDER},
		{Pos{1, -2}, 1, BEETLE},
		{Pos{1, 0}, 0, BEETLE},
		{Pos{1, -3}, 1, QUEEN},
		{Pos{0, 2}, 0, QUEEN},
		{Pos{2, -1}, 1, SPIDER},
		{Pos{2, 0}, 0, ANT},
	}
	board := buildBoard(layout)
	board.BuildDerived()
	// printBoard(board)

	// Beetle on 1,0: shouldn't be able to move to (1,-1),
	// since it would squeeze between pieces.
	want := []Pos{{0, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}}
	beetleMoves := listMovesForPiece(board, BEETLE, Pos{1, 0})
	if !reflect.DeepEqual(want, beetleMoves) {
		t.Errorf("Wanted Beetle moves to be %v, got %v", want, beetleMoves)
	}

	// Beetle on 0,0: can move to any neighboor position, except (1, -1),
	// since it would squeeze between pieces.
	want = []Pos{{-1, -1}, {0, -1}, {-1, 0}, {1, 0}, {0, 1}}
	PosSort(want)
	beetleMoves = listMovesForPiece(board, BEETLE, Pos{0, 0})
	if !reflect.DeepEqual(want, beetleMoves) {
		t.Errorf("Wanted Beetle moves to be %v, got %v", want, beetleMoves)
	}
}

func TestAct(t *testing.T) {
	layout := []PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{0, 0}, 1, BEETLE},
		{Pos{0, 0}, 0, BEETLE},
		{Pos{0, -1}, 1, ANT},
		{Pos{0, 1}, 0, SPIDER},
		{Pos{1, -2}, 1, BEETLE},
		{Pos{1, 0}, 0, BEETLE},
		{Pos{1, -3}, 1, QUEEN},
		{Pos{0, 2}, 0, QUEEN},
		{Pos{2, -1}, 1, SPIDER},
		{Pos{2, 0}, 0, ANT},
	}
	board := buildBoard(layout)
	board.BuildDerived()
	// printBoard(board)

	// Player 0: unstack beetle.
	board = board.Act(Action{Move: true, Piece: BEETLE, SourcePos: Pos{0, 0}, TargetPos: Pos{-1, -1}})
	player, piece, stacked := board.PieceAt(Pos{-1, -1})
	if player != 0 || piece != BEETLE || stacked {
		t.Errorf("Expected Player's 0 Beetle unstacked at (-1, -1), got player=%d, piece=%s, stacked=%v",
			player, PieceNames[piece], stacked)
	}
	player, piece, stacked = board.PieceAt(Pos{0, 0})
	count := board.CountAt(Pos{0, 0})
	if player != 1 || piece != BEETLE || !stacked || count != 2 {
		t.Errorf("Expected Player's 1 Beetle stacked at (0, 0), got player=%d, piece=%s, stacked=%v, count=%d",
			player, PieceNames[piece], stacked, count)
	}

	// Player 1: move beetle.
	board = board.Act(Action{Move: true, Piece: BEETLE, SourcePos: Pos{0, 0}, TargetPos: Pos{0, -1}})
	player, piece, stacked = board.PieceAt(Pos{0, 0})
	if player != 0 || piece != ANT || stacked {
		t.Errorf("Expected Player's 0 Ant stacked at (0, 0), got player=%d, piece=%s, stacked=%v",
			player, PieceNames[piece], stacked)
	}
	player, piece, stacked = board.PieceAt(Pos{0, -1})
	count = board.CountAt(Pos{0, -1})
	if player != 1 || piece != BEETLE || !stacked || count != 2 {
		t.Errorf("Expected Player's 1 Beetle stacked at (0, -1), got player=%d, piece=%s, stacked=%v, count=%d",
			player, PieceNames[piece], stacked, count)
	}
	// printBoard(board)

	// Test situation where there are no action possible.
	layout = []PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{0, 0}, 1, BEETLE},
	}
	board = buildBoard(layout)
	board.BuildDerived()
	if len(board.Derived.Actions) != 0 {
		t.Errorf("Expected no action available, got %v", board.Derived.Actions)
	}
}
