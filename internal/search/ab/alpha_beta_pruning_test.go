package ab_test

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/ai/features"
	features2 "github.com/janpfeifer/hiveGo/internal/features"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"reflect"
	"testing"

	"github.com/janpfeifer/hiveGo/ascii_ui"
)

var _ = fmt.Printf

type PieceLayout struct {
	pos    Pos
	player uint8
	piece  PieceType
}

var scorer = features.TrainedBest

func buildBoard(layout []PieceLayout) (b *Board) {
	b = NewBoard()
	for _, p := range layout {
		b.StackPiece(p.pos, p.player, p.piece)
		b.SetAvailable(p.player, p.piece, b.Available(p.player, p.piece)-1)
	}
	return
}

func listMovesForPiece(b *Board, piece PieceType, pos Pos) (poss []Pos) {
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
	features2.PrettyPrintFeatures(features2.FeatureVector(b, features2.BoardFeaturesDim))
}

func TestEndGameMove(t *testing.T) {
	board := buildBoard([]PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, QUEEN},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, 1}, 0, SPIDER},
		{Pos{-2, 2}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, SPIDER},
		{Pos{-1, 2}, 1, SPIDER},
		{Pos{2, 0}, 0, ANT},
		{Pos{1, -1}, 0, ANT},
	})
	board.NextPlayer = 1
	board.BuildDerived()

	action, _, score := AlphaBeta(board, scorer, 2, false)
	want := Action{Move: true, Piece: GRASSHOPPER, SourcePos: Pos{-2, 2}, TargetPos: Pos{0, 1}}
	if !reflect.DeepEqual(want, action) {
		printBoard(board)
		t.Errorf("Wanted %s, got %s -> score=%.2f\n", want, action, score)
	}
}
