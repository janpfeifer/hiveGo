package features_test

import (
	"fmt"
	featuresLib "github.com/janpfeifer/hiveGo/internal/features"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"github.com/stretchr/testify/assert"
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

func TestNumPlacementPositions(t *testing.T) {
	spec := featuresLib.BoardSpecs[featuresLib.IdNumPlacementPositions]

	// Board with 1 piece each, each will have 3 placement options.
	board := NewBoard()
	board = board.Act(Action{Piece: ANT, TargetPos: Pos{0, 0}}) // Player 0
	board = board.Act(Action{Piece: ANT, TargetPos: Pos{1, 0}}) // Player 1
	printBoard(board)
	board.BuildDerived()
	fmt.Printf("Board feaured dim is %d\n", featuresLib.BoardFeaturesDim)
	features := featuresLib.ForBoard(board, featuresLib.BoardFeaturesDim)
	featuresLib.PrettyPrint(features)
	assert.Equal(t, float32(3), features[spec.VecIndex])
	assert.Equal(t, float32(3), features[spec.VecIndex+1])

	// Asymmetric board: player 0 has 7 placement positions, player 1 has 6.
	board = NewBoard()
	board = board.Act(Action{Piece: ANT, TargetPos: Pos{0, 0}})  // Player 0
	board = board.Act(Action{Piece: ANT, TargetPos: Pos{1, 0}})  // Player 1
	board = board.Act(Action{Piece: ANT, TargetPos: Pos{-1, 0}}) // Player 0
	board = board.Act(Action{Piece: ANT, TargetPos: Pos{2, -1}}) // Player 1
	board = board.Act(Action{Piece: ANT, TargetPos: Pos{-2, 0}}) // Player 0
	board = board.Act(Action{Piece: ANT, TargetPos: Pos{2, 0}})  // Player 1
	printBoard(board)
	board.BuildDerived()
	fmt.Printf("Board feaured dim is %d\n", featuresLib.BoardFeaturesDim)
	features = featuresLib.ForBoard(board, featuresLib.BoardFeaturesDim)
	featuresLib.PrettyPrint(features)
	assert.Equal(t, float32(7), features[spec.VecIndex])
	assert.Equal(t, float32(6), features[spec.VecIndex+1])
}
