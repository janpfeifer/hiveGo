package ai

import (
	. "github.com/janpfeifer/hiveGo/state"
)

// Scorer scores: the convention is that victory/defeat are assigned to
// +10 and -10 respectivelly. Values used on a softmax when used to
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
type BatchScorerWrapper struct {
	Scorer
}

func (s BatchScorerWrapper) BatchScore(boards []*Board) []float64 {
	scores := make([]float64, len(boards))
	for ii, board := range boards {
		scores[ii] = s.Score(board)
	}
	return scores
}

// Returns weather it's the end of the game, and the hard-coded score of a win/loss/draw
// for the current player if it is finished.
func EndGameScore(b *Board) (isEnd bool, score float64) {
	if !b.IsFinished() {
		return false, 0
	}
	if b.Draw() {
		return true, 0
	}
	if b.Derived.Wins[b.NextPlayer] {
		// Current player wins.
		return true, 10.0
	}
	// Opponent player wins.
	return true, -10.0
}
