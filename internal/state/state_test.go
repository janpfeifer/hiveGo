package state_test

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"github.com/stretchr/testify/assert"
	"testing"
)

var _ = fmt.Printf

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

// listMovesForPieceDisplayPos takes the piece position in "display coordinates", and returns
// the moves also in "display coordinates".
func listMovesForPieceDisplayPos(b *Board, piece PieceType, displayPos Pos) []Pos {
	pos := displayPos.FromDisplayPos()
	moves := listMovesForPiece(b, piece, pos)
	for idx := range moves {
		moves[idx] = moves[idx].ToDisplayPos()
	}
	return moves
}

func listMovesForPiece(b *Board, piece PieceType, pos Pos) []Pos {
	var moves []Pos
	d := b.Derived
	for _, a := range d.Actions {
		if a.Move && a.SourcePos == pos && a.Piece == piece {
			moves = append(moves, a.TargetPos)
		}
	}
	SortPositions(moves)
	return moves
}

func printBoard(b *Board) {
	ui := cli.New(true, false)
	ui.PrintBoard(b)
}

func TestEqual(t *testing.T) {
	a1 := Action{Piece: NoPiece}
	a2 := Action{Piece: NoPiece, Move: true}
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
		{0, 0}, {1, 0}, {-1, 0}, {2, 0},
		{0, 5}, {1, -5}, {-1, 7}, {3, -7},
	}
	want := []Pos{
		{0, 0}, {1, 0}, {-1, -1}, {2, 1},
		{0, 5}, {1, -5}, {-1, 6}, {3, -6},
	}
	for idx, pos := range from {
		if got := pos.ToDisplayPos(); got != want[idx] {
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

func TestRemovablePositions(t *testing.T) {
	board := buildBoard([]PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, SPIDER},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 0}, 0, QUEEN},
		{Pos{-1, 2}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, SPIDER},
		{Pos{-1, 3}, 1, SPIDER},
	}, false)
	printBoard(board)
	board.BuildDerived()

	want := generics.SetWith(Pos{-1, 3}, Pos{-1, 0}, Pos{2, 0}, Pos{1, 1})
	assert.Equal(t, want, board.Derived.RemovablePositions)
}

func TestAct(t *testing.T) {
	layout := []PieceOnBoard{
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
	layout = []PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{0, 0}, 1, BEETLE},
	}
	board = buildBoard(layout, true)
	board.BuildDerived()
	if len(board.Derived.Actions) != 1 || board.Derived.Actions[0] != SkipAction {
		t.Errorf("Expected SkipAction (pass action), got %v", board.Derived.Actions)
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

// TestInvalidMove is just a placeholder used during development.
func TestInvalidMove(t *testing.T) {
	b := NewBoard()
	actions := []Action{
		{Move: false, Piece: BEETLE, TargetPos: Pos{0, 0}},
		{Move: false, Piece: QUEEN, TargetPos: Pos{0, 1}},

		{Move: false, Piece: BEETLE, TargetPos: Pos{-1, 0}},
		{Move: false, Piece: ANT, TargetPos: Pos{0, 2}},

		{Move: false, Piece: QUEEN, TargetPos: Pos{1, -1}},
		{Move: true, Piece: ANT, SourcePos: Pos{0, 2}, TargetPos: Pos{1, -1}},

		{Move: false, Piece: ANT, TargetPos: Pos{-2, 1}},
		{Move: false, Piece: BEETLE, TargetPos: Pos{3, -1}},

		{Move: true, Piece: ANT, SourcePos: Pos{-2, 1}, TargetPos: Pos{1, 1}},
		{Move: true, Piece: BEETLE, SourcePos: Pos{3, -1}, TargetPos: Pos{2, 0}},

		{Move: true, Piece: QUEEN, SourcePos: Pos{1, -1}, TargetPos: Pos{2, -2}},
	}
	for ii, act := range actions {
		b = b.Act(act)
		fmt.Printf("Move %d (ii=%d), %s, Player %d, Repeats: %d, Hash: %x\n",
			b.MoveNumber, ii, act, b.NextPlayer, b.Derived.Repeats, b.Derived.Hash)
	}
	printBoard(b)
}

func BenchmarkCalcDerived(b *testing.B) {
	layout := []PieceOnBoard{
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		board.Act(action)
	}
}
