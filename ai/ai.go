package ai

import (
	. "github.com/janpfeifer/hiveGo/state"
)

// Scorer scores: the convention is that victory/defeat are assigned to
// +100 and -100 respectivelly. Values used on a softmax when used to
// determine probabilistic moves.
type Scorer interface {
	Score(board *Board) float64
}

// BatchScorer is a Scorer that also handles batches.
type BatchScorer interface {
	Scorer

	// BatchScore aggregate scoring in batches -- presumable more efficient.
	BatchScore(boards []*Board) []float64
}

// Trivial implementation of a BatchScorer, wity no efficiency gains.
type WrapperBatchScorer struct {
	Scorer
}

func (s WrapperBatchScorer) BatchScore(boards []*Board) []float64 {
	scores := make([]float64, len(boards))
	for ii, board := range boards {
		scores[ii] = s.Score(board)
	}
	return scores
}

// Returns weather it's the end of the game, and the hard-coded score of a win/loss/draw
// for the current player if it is finished.
func EndGameScore(b *Board) (isEnd bool, score float64) {
	if !b.Derived.Wins[0] && !b.Derived.Wins[0] {
		return false, 0
	}
	if b.Derived.Wins[0] {
		if b.Derived.Wins[1] {
			// Draw.
			return true, 0
		}
		// Current player wins.
		return true, 100.0
	}
	// Opponent player wins.
	return true, -100.0
}
