package ai

import (
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
)

// BatchBoardScorerProxy is a trivial implementation of a BatchBoardScorer, with no efficiency gains.
type BatchBoardScorerProxy struct {
	BoardScorer
}

// BatchBoardScore calls the BoardScore for each board of the batch.
func (s BatchBoardScorerProxy) BatchBoardScore(boards []*Board) (scores []float32) {
	scores = generics.SliceMap(boards, func(board *Board) float32 {
		return s.BoardScore(board)
	})
	return
}

func (s BatchBoardScorerProxy) String() string {
	return s.BoardScorer.String()
}

// Assert BatchBoardScorerProxy implements BatchBoardScorer
var _ BatchBoardScorer = &BatchBoardScorerProxy{}
