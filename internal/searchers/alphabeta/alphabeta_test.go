package alphabeta_test

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai/linear"
	features2 "github.com/janpfeifer/hiveGo/internal/features"
	"github.com/janpfeifer/hiveGo/internal/searchers/alphabeta"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"
	"testing"

	"github.com/janpfeifer/hiveGo/internal/ui/cli"
)

var _ = fmt.Printf

type PieceLayout struct {
	pos    Pos
	player PlayerNum
	piece  PieceType
}

var scorer = linear.PreTrainedBest

func init() {
	klog.InitFlags(nil)
}

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
	ui := cli.New(true, false)
	ui.PrintBoard(b)
	features2.PrettyPrint(features2.ForBoard(b, features2.BoardFeaturesDim))
}

func TestEndGameMove(t *testing.T) {
	board := buildBoard([]PieceLayout{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, QUEEN},
		{Pos{-1, 1}, 1, QUEEN},
		{Pos{2, -1}, 0, SPIDER},
		{Pos{-2, 3}, 1, GRASSHOPPER},
		{Pos{1, 1}, 0, SPIDER},
		{Pos{-1, 2}, 1, SPIDER},
		{Pos{2, 0}, 0, ANT},
		{Pos{1, -1}, 0, ANT},
	})
	board.NextPlayer = 1
	board.BuildDerived()
	printBoard(board)

	searcher := alphabeta.New(scorer).WithMaxDepth(1)
	action, _, score, _ := searcher.Search(board)
	want := Action{Move: true, Piece: GRASSHOPPER, SourcePos: Pos{-2, 3}, TargetPos: Pos{0, 1}}
	assert.Equalf(t, want, action, "Wanted %s, got %s -> score=%.2f", want, action, score)
}

func TestTwoMovesMove(t *testing.T) {
	board := buildBoard([]PieceLayout{
		{Pos{0, 0}, 0, QUEEN},
		{Pos{1, 0}, 0, ANT},
		{Pos{2, 0}, 0, ANT},
		{Pos{3, 0}, 0, ANT},
		{Pos{4, 0}, 0, BEETLE},

		{Pos{-1, 1}, 1, BEETLE},
		{Pos{0, -1}, 1, SPIDER},
		{Pos{-1, 0}, 1, SPIDER},
		{Pos{0, 1}, 1, ANT},
		{Pos{-2, 1}, 1, QUEEN},
		{Pos{-2, 0}, 1, GRASSHOPPER},
	})
	board.NextPlayer = 1
	board.SetAvailable(1, ANT, 0)
	board.SetAvailable(1, BEETLE, 0)
	board.SetAvailable(1, SPIDER, 0)
	board.SetAvailable(1, GRASSHOPPER, 0)
	board.BuildDerived()
	printBoard(board)

	searcher := alphabeta.New(scorer).WithMaxDepth(3)
	action, _, score, _ := searcher.Search(board)
	want := Action{Move: true, Piece: GRASSHOPPER, SourcePos: Pos{-2, 0}, TargetPos: Pos{-2, 2}}
	assert.Equalf(t, want, action, "Wanted %s, got %s -> score=%.2f", want, action, score)
}
