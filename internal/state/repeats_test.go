package state_test

import (
	"fmt"
	. "github.com/janpfeifer/hiveGo/internal/state"
	. "github.com/janpfeifer/hiveGo/internal/state/statetest"
	"testing"
)

func checkDraw(t *testing.T, b *Board, draw bool) {
	if b.Draw() != draw {
		t.Errorf("TestRepeats: board at move number %d wanted draw=%v, got draw=%v, repeats=%d",
			b.MoveNumber, draw, !draw, b.Derived.Repeats)
		PrintBoard(b)
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
	PrintBoard(b)
	fmt.Println()

	for ii := int8(0); ii < 6; ii++ {
		b = b.Act(Action{Move: true, Piece: QUEEN, SourcePos: Pos{ii, 0}, TargetPos: Pos{ii + 1, 0}})
		fmt.Printf("Move %d (ii=%d), Player %d, Repeats: %d, Hash: %x\n",
			b.MoveNumber, ii, b.NextPlayer, b.Derived.Repeats, b.Derived.Hash)
		PrintBoard(b)
		fmt.Println()
		checkDraw(t, b, false)

		// At the last repeat this position will be repeating the third time.
		b = b.Act(Action{Move: true, Piece: QUEEN, SourcePos: Pos{ii, 1}, TargetPos: Pos{ii + 1, 1}})
		fmt.Printf("Move %d (ii=%d), Player %d, Repeats: %d, Hash: %x\n",
			b.MoveNumber, ii, b.NextPlayer, b.Derived.Repeats, b.Derived.Hash)
		PrintBoard(b)
		fmt.Println()
		checkDraw(t, b, ii == 5)
	}

	// Check that another move of the first player also repeats.
	ii := int8(6)
	b = b.Act(Action{Move: true, Piece: QUEEN, SourcePos: Pos{ii, 0}, TargetPos: Pos{ii + 1, 0}})
	fmt.Printf("Move %d (ii=%d), Player %d, Repeats: %d, Hash: %x\n",
		b.MoveNumber, ii, b.NextPlayer, b.Derived.Repeats, b.Derived.Hash)
	PrintBoard(b)
	fmt.Println()
	checkDraw(t, b, true)

	// Finally a placement should break the repeats.
	b = b.Act(Action{Move: false, Piece: ANT, TargetPos: Pos{6, 0}})
	checkDraw(t, b, false)
}
