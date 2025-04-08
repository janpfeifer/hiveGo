package ai

import (
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
)

// BatchBoardScorerProxy is a trivial implementation of a BatchValueScorer, with no efficiency gains.
type BatchBoardScorerProxy struct {
	ValueScorer
}

// BatchBoardScore calls the Score for each board of the batch.
func (s BatchBoardScorerProxy) BatchScore(boards []*Board) (scores []float32) {
	scores = generics.SliceMap(boards, func(board *Board) float32 {
		return s.Score(board)
	})
	return
}

func (s BatchBoardScorerProxy) String() string {
	return s.ValueScorer.String()
}

// Assert BatchBoardScorerProxy implements BatchValueScorer
var _ BatchValueScorer = &BatchBoardScorerProxy{}
