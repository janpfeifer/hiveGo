package searchers

import (
	"github.com/gomlx/exceptions"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"math"
	"math/rand/v2"
	"slices"
)

// NewRandomizedSearcher adds randomness to the action taken by an existing Searcher.
// Args:
//
//   - searcher: Baseline Searcher.
//   - randomness (>=0): Amount of randomness to use: it is applied as a divisor to the scores
//     returned by the Searcher, except if there is a winning move.
//     The larger the value the more it leads to randomness (exploration), and lower values
//     lead to "pick the best scoring move" (exploitation), with zero meaning no randomness.
//   - maxMoveRandomness: starting at this move no more randomness is used. This allows
//     randomness to be used only earlier in the match.
func NewRandomizedSearcher(searcher Searcher, randomness float64, maxMoveRandomness int) Searcher {
	if randomness <= 0 {
		// Without randomness, simply return the original Searcher.
		return searcher
	}
	return &randomizedSearcher{searcher: searcher, randomness: randomness, maxMoveRandomness: maxMoveRandomness}
}

// randomizedSearcher is a meta Searcher, that introduces randomness to its scorer.
type randomizedSearcher struct {
	searcher          Searcher
	randomness        float64
	maxMoveRandomness int
}

// Assert randomizedSearcher is a Searcher.
var _ Searcher = &randomizedSearcher{}

// Search implements the Searcher interface.
func (rs *randomizedSearcher) Search(board *Board) (chosenAction Action, nextBoard *Board, score float32, actionsScores []float32) {
	actions := board.Derived.Actions

	// Get scores from base searcher for current board.
	chosenAction, nextBoard, score, actionsScores = rs.searcher.Search(board)

	// If we reached the max move number for randomness, or if the searcher doesn't return scores for the
	// different actions, or if there is only one action possible, or if it is an end-game move,
	// we don't add any randomness.
	if board.MoveNumber >= rs.maxMoveRandomness || nextBoard.IsFinished() || len(actionsScores) <= 1 {
		return
	}
	if len(actionsScores) != len(actions) {
		exceptions.Panicf("randomizedSearcher: Searcher returned %d actionsScores, but board has %d actions!?", len(actionsScores), len(actions))
	}

	// Calculate probability for each action.
	logits := make([]float64, len(actionsScores))
	for ii, score := range actionsScores {
		logits[ii] = float64(score) / rs.randomness
	}
	probabilities := softmax(logits)

	// Select from probabilities.
	chance := rand.Float64()
	// klog.Infof("chance=%f, scores=%v, probabilities=%v", chance, scores, probabilities)
	for actionIdx, value := range probabilities {
		if chance > value {
			chance -= value
			continue
		}

		// Found the new action:
		if klog.V(2).Enabled() {
			klog.Infof("randomizedSearcher selection: action=%s, score=%s", actions[actionIdx], actionsScores[actionIdx])
		}
		if actions[actionIdx] == chosenAction {
			// randomizedSearcher chose the same as the base searcher.
			return
		}
		chosenAction = actions[actionIdx]
		nextBoard = board.Act(chosenAction)
		score = actionsScores[actionIdx]
		return
	}
	// It should not reach here.
	exceptions.Panicf("Nothing selected!? remaining chance=%f, probabilities=%v", chance, probabilities)
	return
}

func softmax(values []float64) (probs []float64) {
	probs = make([]float64, len(values))
	var sum float64

	// Subtract maxValue from all values keep the probability the same, but makes for more numerically stable
	// values.
	maxValue := slices.Max(values)
	// Normalize value for numeric values (smaller exponentials)
	for ii, value := range values {
		probs[ii] = math.Exp(value - maxValue)
		sum += probs[ii]
	}
	for ii := range probs {
		probs[ii] /= sum
	}
	return
}
