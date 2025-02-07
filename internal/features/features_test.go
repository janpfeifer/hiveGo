package features_test

import (
	"fmt"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"testing"
)

func printBoard(b *Board) {
	fmt.Println()
	ui := cli.New(true, false)
	ui.PrintBoard(b)
	fmt.Println()
}

func TestAverageDistances(t *testing.T) {
	b := NewBoard()
	b.BuildDerived()
	b = b.Act(Action{false, ANT, Pos{0, 0}, Pos{}})   // Player 0
	b = b.Act(Action{false, ANT, Pos{1, 0}, Pos{}})   // Player 1
	b = b.Act(Action{false, ANT, Pos{0, 1}, Pos{}})   // Player 0
	b = b.Act(Action{false, ANT, Pos{2, 0}, Pos{}})   // Player 1
	b = b.Act(Action{false, QUEEN, Pos{0, 2}, Pos{}}) // Player 0
	b = b.Act(Action{false, QUEEN, Pos{3, 0}, Pos{}}) // Player 1
}
