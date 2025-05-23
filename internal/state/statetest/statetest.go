// Package statetest provides helper functions to create tests using Hive state.
package statetest

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
)

// PieceOnBoard represents a position and ownership of a piece in the board.
type PieceOnBoard struct {
	Pos    Pos
	Player PlayerNum
	Piece  PieceType
}

func PrintBoard(b *Board) {
	ui := cli.New(true, false)
	ui.PrintBoard(b)
}

func PrintActions(board *Board, actionTaken Action, policy []float32) {
	ui := cli.New(true, false)
	ui.PrettyPrintActionsWithPolicy(board, policy, actionTaken, 5)
}

// BuildBoard from a collection of pieces. Their positions may be in "display coordinates".
func BuildBoard(layout []PieceOnBoard, displayPos bool) (b *Board) {
	b = NewBoard()
	for _, p := range layout {
		pos := p.Pos
		if displayPos {
			pos = pos.FromDisplayPos()
		}
		b.StackPiece(pos, p.Player, p.Piece)
		b.SetAvailable(p.Player, p.Piece, b.Available(p.Player, p.Piece)-1)
	}
	return
}
