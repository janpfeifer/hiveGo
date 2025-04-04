package ai

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
	"math"
	"slices"
)

// PolicyProxy implement a PolicyScorer that wraps a common BoardScorer.
// It scores the policy by using the score of the state of each action taken.
//
// It allows the scorer to work with MCTS (Monte Carlo Tree Search) searcher.
//
// It is not a PolicyLearner though, that requires a proper PolicyScorer model.
type PolicyProxy struct {
	BoardScorer
	batchScorer BatchBoardScorer
	scale       float32
}

// NewPolicyProxy returns a proxy PolicyScorer that takes a BoardScorer to score the board states
// for each action, passes the output to a Softmax and return that probability as a policy.
// It also takes scale as a multiplier before the Softmax.
//
// It allows the scorer to work with MCTS (Monte Carlo Tree Search) searcher.
//
// It is not a PolicyLearner though, that requires a proper PolicyScorer model.
func NewPolicyProxy(scorer BoardScorer, scale float32) PolicyScorer {
	p := &PolicyProxy{
		BoardScorer: scorer,
		scale:       scale,
	}
	if batchScorer, ok := scorer.(BatchBoardScorer); ok {
		p.batchScorer = batchScorer
	} else {
		p.batchScorer = BatchBoardScorerProxy{scorer}
	}
	return p
}

// PolicyScore implements PolicyScorer.
func (p *PolicyProxy) PolicyScore(board *Board) []float32 {
	nextBoards := board.TakeAllActions()
	scores := p.batchScorer.BatchBoardScore(nextBoards)
	if p.scale != 1 {
		for ii := range scores {
			scores[ii] = p.scale * scores[ii]
		}
	}
	return Softmax(scores)
}

// Softmax returns the Softmax of the given logits in a numerically stable way.
func Softmax(logits []float32) (probs []float32) {
	probs = make([]float32, len(logits))
	var sum float32

	// Subtract maxValue from all logits keep the probability the same, but makes for more numerically stable
	// logits.
	maxValue := slices.Max(logits)
	// Normalize value for numeric logits (smaller exponentials)
	for ii, value := range logits {
		probs[ii] = float32(math.Exp(float64(value - maxValue)))
		sum += probs[ii]
	}
	for ii := range probs {
		probs[ii] /= sum
	}
	return
}
