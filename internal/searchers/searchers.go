package searchers

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
	"math"
	"slices"
)

var (
	// IdleChan if created is read before each chunk of search is done.
	// This allows for processing to happen in the idle callbacks when running in javascript (via GopherJS).
	IdleChan chan bool
)

// Searcher is the interface that any of the search algorithms
// must adhere to be valid.
type Searcher interface {
	// Search returns the next action to take on the given board, along with the updated Board (after taking the action)
	// and the expected score of taking that action.
	Search(board *Board) (bestAction Action, bestBoard *Board, bestScore float32, err error)
}

// softmax returns the softmax of the given logits in a numerically stable way.
func softmax(logits []float64) (probs []float64) {
	probs = make([]float64, len(logits))
	var sum float64

	// Subtract maxValue from all logits keep the probability the same, but makes for more numerically stable
	// logits.
	maxValue := slices.Max(logits)
	// Normalize value for numeric logits (smaller exponentials)
	for ii, value := range logits {
		probs[ii] = math.Exp(value - maxValue)
		sum += probs[ii]
	}
	for ii := range probs {
		probs[ii] /= sum
	}
	return
}
