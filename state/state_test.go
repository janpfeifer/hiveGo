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

func PieceLayoutsFromDisplayPos(pls []PieceLayout) {
	for ii := range pls {
		pls[ii].pos = pls[ii].pos.FromDisplayPos()
	}
}

func buildBoard(layout []PieceLayout, displayPos bool) (b *Board) {
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

func listMovesForPieceDisplayPos(b *Board, piece Piece, pos Pos) (poss []Pos) {
	pos = pos.FromDisplayPos()
	poss = listMovesForPiece(b, piece, pos)
	PosSlice(poss).DisplayPos()
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

func TestEqual(t *testing.T) {
	a1 := Action{Piece: NO_PIECE}
	a2 := Action{Piece: NO_PIECE, Move: true}
	if !a1.Equal(a2) {
		t.Errorf("Expected %s and %s to be the same.", a1, a2)
	}

	a2 = Action{Piece: ANT}
	if a1.Equal(a2) {
		t.Errorf("Expected %s and %s to be different.", a1, a2)
	}

	a1 = Action{Piece: BEETLE, TargetPos: Pos{1, -1}}
	a2 = Action{Piece: BEETLE, TargetPos: Pos{1, -1}, SourcePos: Pos{10, 20}}
	if !a1.Equal(a2) {
		t.Errorf("Expected %s and %s to be the same.", a1, a2)
	}

	a2 = Action{Piece: BEETLE, TargetPos: Pos{1, -2}}
	if a1.Equal(a2) {
		t.Errorf("Expected %s and %s to be different.", a1, a2)
	}

	a2 = Action{Piece: BEETLE, TargetPos: Pos{1, -1}, Move: true}
	if a1.Equal(a2) {
		t.Errorf("Expected %s and %s to be different.", a1, a2)
	}

	a1 = Action{Move: true, Piece: GRASSHOPPER, SourcePos: Pos{10, -10}, TargetPos: Pos{1, -1}}
	a2 = Action{Move: true, Piece: GRASSHOPPER, SourcePos: Pos{10, -10}, TargetPos: Pos{1, -1}}
	if !a1.Equal(a2) {
		t.Errorf("Expected %s and %s to be the same.", a1, a2)
	}

	a2.SourcePos = Pos{7, 7}
	if a1.Equal(a2) {
		t.Errorf("Expected %s and %s to be different.", a1, a2)
	}
}

func TestDisplayPos(t *testing.T) {
	// Convert to display positions.
	from := []Pos{
		Pos{0, 0}, Pos{1,0}, Pos{-1,0}, Pos{2, 0},
		Pos{0, 5}, Pos{1,-5}, Pos{-1,7}, Pos{3, -7},
	}
	want := []Pos{
		Pos{0, 0}, Pos{1,0}, Pos{-1,-1}, Pos{2, 1},		
		Pos{0, 5}, Pos{1,-5}, Pos{-1,6}, Pos{3, -6},
	}
	for idx, pos := range from {
		if got := pos.DisplayPos(); got != want[idx] {
			t.Errorf("Convert position to display: pos=%s, got=%s, wanted=%s", pos, got, want[idx])
		}
	}

	// Convert from display position.
	from, want = want, from
	for idx, pos := range from {
		if got := pos.FromDisplayPos(); got != want[idx] {
			t.Errorf("Convert from display position: pos=%s, got=%s, wanted=%s", pos, got, want[idx])
		}
	}
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
	}, true)
	board.BuildDerived()

	displayPosWant := map[Pos]bool{Pos{-1, 3}: true, Pos{1, 1}: true, Pos{2, 1}: true}
	want := make(map[Pos]bool)
	for displayPos := range displayPosWant {
		want[displayPos.FromDisplayPos()] = true 
	}

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
	board := buildBoard(layout, true)
	board.BuildDerived()
	printBoard(board)

	queenMoves := listMovesForPiece(board, QUEEN, Pos{2, 0})
	want := []Pos{{2, -1}, {1, 1}}
	if !reflect.DeepEqual(want, queenMoves) {
		t.Errorf("Wanted Queen moves to be %v, got %v", want, queenMoves)
	}

	// Now queen can't move because it would break hive.
	layout = append(layout, PieceLayout{Pos{3, 1}, 0, GRASSHOPPER})
	board = buildBoard(layout, true)
	printBoard(board)
	board.BuildDerived()
	queenMoves = listMovesForPiece(board, QUEEN, Pos{2, 0})
	if len(queenMoves) != 0 {
		t.Errorf("Wanted Queen moves to be empty, got %v", queenMoves)
	}

	// If we put some pieces around it can move again.
	layout = append(layout, PieceLayout{Pos{1, 1}, 0, BEETLE})
	layout = append(layout, PieceLayout{Pos{2, 2}, 0, BEETLE})
	board = buildBoard(layout, true)
	printBoard(board)
	board.BuildDerived()
	queenMoves = listMovesForPiece(board, QUEEN, Pos{2, 0})
	want = []Pos{{2, -1}, {3, -1}}
	if !reflect.DeepEqual(want, queenMoves) {
		t.Errorf("Wanted Queen moves to be %v, got %v", want, queenMoves)
	}

	// Finally piece is not supposed to squeeze among pieces:
	layout = append(layout, PieceLayout{Pos{2, 0}, 0, ANT})
	board = buildBoard(layout, true)
	board.BuildDerived()
	queenMoves = listMovesForPieceDisplayPos(board, QUEEN, Pos{2, 1})
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
	board := buildBoard(layout, true)
	printBoard(board)
	board.BuildDerived()

	// Spider at (1,0) should be locked in place (it would break the hive)
	spiderMoves := listMovesForPieceDisplayPos(board, SPIDER, Pos{1, 0})
	if len(spiderMoves) != 0 {
		t.Errorf("Wanted Spider moves to be empty, got %v", spiderMoves)
	}

	// Spider at (1,1) is free to move.
	want := []Pos{{3, 0}, {0, 3}}
	spiderMoves = listMovesForPieceDisplayPos(board, SPIDER, Pos{1, 1})
	if !reflect.DeepEqual(want, spiderMoves) {
		PosSlice(want).FromDisplayPos()
		//PosSlice(spiderMoves).FromDisplayPos()
		t.Errorf("Wanted Spider moves to be %v, got %v", want, spiderMoves)
	}

	// Spider at (-1, 3)
	want = []Pos{{-2, 1}, {1, 2}}
	// printBoard(board)
	spiderMoves = listMovesForPieceDisplayPos(board, SPIDER, Pos{-1, 3})
	if !reflect.DeepEqual(want, spiderMoves) {
		PosSlice(want).FromDisplayPos()
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
	board := buildBoard(layout, true)
	board.BuildDerived()
	// printBoard(board)

	// Grasshoppers at (2,1) and (1,1) can jump to 2 directions.
	want := []Pos{{1, -1}, {3, 0}}
	grasshopperMoves := listMovesForPieceDisplayPos(board, GRASSHOPPER, Pos{1, 1})
	if !reflect.DeepEqual(want, grasshopperMoves) {
		t.Errorf("Wanted grasshopper moves to be %v, got %v", want, grasshopperMoves)
	}
	want = []Pos{{-1, -1}, {0, 2}}
	grasshopperMoves = listMovesForPieceDisplayPos(board, GRASSHOPPER, Pos{2, 1})
	if !reflect.DeepEqual(want, grasshopperMoves) {
		t.Errorf("Wanted grasshopper moves to be %v, got %v", want, grasshopperMoves)
	}

	grasshopperMoves = listMovesForPieceDisplayPos(board, GRASSHOPPER, Pos{-1, 2})
	if len(grasshopperMoves) != 0 {
		t.Errorf("Wanted grasshopper moves to be empty, got %v", grasshopperMoves)
	}
}

func TestAntMoves(t *testing.T) {
	layout := []PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 1}, 1, BEETLE},
		{Pos{1, 0}, 0, QUEEN},
		{Pos{-1, 2}, 1, QUEEN},
		{Pos{2, 0}, 0, ANT},
		{Pos{-1, 3}, 1, ANT},
		{Pos{1, 1}, 0, ANT},
		{Pos{-1, 4}, 1, ANT},
	}
	board := buildBoard(layout, false)
	board.BuildDerived()
	printBoard(board)

	// Ant at (1,1) can move anywhere connected to the hive.
	want := []Pos{{0, -1}, {1, -1}, {2, -1}, {3, -1}, {-1, 0}, {3, 0}, {0, 1}, {2, 1}, {-2, 1}, {-2, 2}, {0, 2}, {-2, 3},
		{0, 3}, {-2, 4}, {0, 4}, {-2, 5}, {-1, 5}}
	PosSort(want)
	antMoves := listMovesForPiece(board, ANT, Pos{1, 1})
	if !reflect.DeepEqual(want, antMoves) {
		t.Errorf("Wanted Ant moves to be:\n%v, got\n%v", want, antMoves)
	}

	// Ant at (2,0) can't squeeze between pieces into (0,1), but anywhere
	// connected to the hive should be fine.
	//want = []Pos{{-2, 0}, {-2, 1}, {-2, 2}, {-2, 3}, {-2, 4}, {-1, -1}, {-1, 4},
	//	{0, -1}, {0, 2}, {0, 3}, {0, 4}, {1, -1}, {1, 2}, {2, 0}, {2, 2}}
	//PosSlice(want).FromDisplayPos()
	want = []Pos{{0, -1}, {1, -1}, {2, -1}, {2, 1}, {1, 2}, {0, 2}, {0, 3}, {0, 4}, {-1, 5},
		{-2, 5}, {-2, 4}, {-2, 3}, {-2, 2}, {-2, 1}, {-1, 0}}
	PosSort(want)
	antMoves = listMovesForPiece(board, ANT, Pos{2, 0})
	if !reflect.DeepEqual(want, antMoves) {
		t.Errorf("Wanted Ant moves to be:\n%v, got\n%v", want, antMoves)
	}

	// Ant at (-1,2) should be blocked.
	antMoves = listMovesForPieceDisplayPos(board, ANT, Pos{-1, 2})
	if len(antMoves) != 0 {
		t.Errorf("Wanted Ant moves to be empty, got\n%v", antMoves)
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
	board := buildBoard(layout, true)
	printBoard(board)
	board.BuildDerived()

	// Beetle on 1,0: shouldn't be able to move to (1,-1),
	// since it would squeeze between pieces.
	want := []Pos{{0, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}}
	PosSort(want)
	beetleMoves := listMovesForPieceDisplayPos(board, BEETLE, Pos{1, 0})
	PosSort(beetleMoves)
	if !reflect.DeepEqual(want, beetleMoves) {
		t.Errorf("Wanted Beetle moves to be\n%v, got\n%v", want, beetleMoves)
	}

	// Beetle on 0,0: can move to any neighboor position, except (1, -1),
	// since it would squeeze between pieces.
	want = []Pos{{-1, -1}, {0, -1}, {-1, 0}, {1, 0}, {0, 1}}
	PosSort(want)
	beetleMoves = listMovesForPieceDisplayPos(board, BEETLE, Pos{0, 0})
	PosSort(beetleMoves)
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
	board := buildBoard(layout, true)
	printBoard(board)
	board.BuildDerived()

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
	board = buildBoard(layout, true)
	board.BuildDerived()
	if len(board.Derived.Actions) != 0 {
		t.Errorf("Expected no action available, got %v", board.Derived.Actions)
	}
}

func checkDraw(t *testing.T, b *Board, draw bool) {
	if b.Draw() != draw {
		t.Errorf("TestRepeats: board at move number %d wanted draw=%v, got draw=%v, repeats=%d",
			b.MoveNumber, draw, !draw, b.Derived.Repeats)
		printBoard(b)
	}
}

// TestRepeats tests that 3 times repeated Board positions get marked as
// a draw.
func TestRepeats(t *testing.T) {
	b := NewBoard()
	b = b.Act(Action{Move: false, Piece: QUEEN, TargetPos: Pos{0, 0}})
	checkDraw(t, b, false)
	b = b.Act(Action{Move: false, Piece: QUEEN, TargetPos: Pos{0, 1}})
	checkDraw(t, b, false)
	printBoard(b)
	fmt.Println()

	for ii := int8(0); ii < 6; ii++ {
		b = b.Act(Action{Move: true, Piece: QUEEN, SourcePos: Pos{ii, 0}, TargetPos: Pos{ii + 1, 0}})
		fmt.Printf("Move %d (ii=%d), Player %d, Repeats: %d, Hash: %x\n",
			b.MoveNumber, ii, b.NextPlayer, b.Derived.Repeats, b.Derived.Hash)
		printBoard(b)
		fmt.Println()
		checkDraw(t, b, false)

		// At the last repeat this position will be repeating the third time.
		b = b.Act(Action{Move: true, Piece: QUEEN, SourcePos: Pos{ii, 1}, TargetPos: Pos{ii + 1, 1}})
		fmt.Printf("Move %d (ii=%d), Player %d, Repeats: %d, Hash: %x\n",
			b.MoveNumber, ii, b.NextPlayer, b.Derived.Repeats, b.Derived.Hash)
		printBoard(b)
		fmt.Println()
		checkDraw(t, b, ii == 5)
	}

	// Check that another move of the first player also repeats.
	ii := int8(6)
	b = b.Act(Action{Move: true, Piece: QUEEN, SourcePos: Pos{ii, 0}, TargetPos: Pos{ii + 1, 0}})
	fmt.Printf("Move %d (ii=%d), Player %d, Repeats: %d, Hash: %x\n",
		b.MoveNumber, ii, b.NextPlayer, b.Derived.Repeats, b.Derived.Hash)
	printBoard(b)
	fmt.Println()
	checkDraw(t, b, true)

	// Finally a placement should break the repeats.
	b = b.Act(Action{Move: false, Piece: ANT, TargetPos: Pos{6, 0}})
	checkDraw(t, b, false)
}

func BenchmarkCalcDerived(b *testing.B) {
	layout := []PieceLayout{
		{Pos{-2, -1}, 1, ANT},
		{Pos{-1, -1}, 0, GRASSHOPPER},
		{Pos{-1, 1}, 0, BEETLE},
		{Pos{-1, 2}, 0, GRASSHOPPER},
		{Pos{0, 0}, 0, BEETLE},
		{Pos{0, 1}, 1, QUEEN},
		{Pos{0, 2}, 0, ANT},
		{Pos{1, -2}, 0, ANT},
		{Pos{1, -1}, 0, SPIDER},
		{Pos{2, 0}, 1, ANT},
		{Pos{2, 1}, 1, GRASSHOPPER},
		{Pos{3, -1}, 1, GRASSHOPPER},
		{Pos{3, 0}, 1, SPIDER},
		{Pos{4, 0}, 0, SPIDER},
		{Pos{5, -1}, 0, QUEEN},
		{Pos{6, 0}, 1, SPIDER},
		{Pos{7, -1}, 1, ANT},
		{Pos{7, 0}, 1, GRASSHOPPER},
	}
	board := buildBoard(layout, true)
	board.BuildDerived()
	board.NextPlayer = 1
	action := Action{SourcePos: Pos{7, -1}, TargetPos: Pos{1, 1}, Piece: ANT}
	printBoard(board)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		board.Act(action)
	}
}
