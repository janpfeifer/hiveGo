package search

import (
	"log"
	"math"
	"math/rand"

	"github.com/janpfeifer/hiveGo/ai"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf

// Searcher is a the interface that any of the search algoriactionthms
// must adhere to be valid.
type Searcher interface {
	// Search returns the next action to take on the given board,
	// along with the updated Board (after taking the action) and
	// the expected score of taking that action.
	Search(b *Board, scorer ai.BatchScorer) (action Action, board *Board, score float64)
}

type randomizedSearcher struct {
	searcher   Searcher
	randomness float64
}

// Search implements the Searcher interface.
func (rs *randomizedSearcher) Search(b *Board, scorer ai.BatchScorer) (Action, *Board, float64) {
	// If there are no valid actions, create the "pass" action
	actions := b.Derived.Actions
	if len(actions) == 0 {
		actions = append(actions, Action{Piece: NO_PIECE})
	}
	scores := make([]float64, len(actions))
	newBoards := make([]*Board, len(actions))

	for ii, action := range actions {
		newBoards[ii] = b.Act(action)
		if isEnd, score := ai.EndGameScore(newBoards[ii]); isEnd {
			// End game is treated differently.
			if score > 0.0 {
				// Player wins, take action (non-randomized)
				return action, newBoards[ii], score
			}
			scores[ii] = score
		} else {
			_, _, scores[ii] = rs.searcher.Search(newBoards[ii], scorer)
		}
	}

	// Special case: randomness == 0 (or less): just take the max.
	if rs.randomness <= 0 {
		maxIdx, maxScore := 0, scores[0]
		for ii := 1; ii < len(scores); ii++ {
			if scores[ii] > maxScore {
				maxScore = scores[ii]
				maxIdx = ii
			}
		}
		return actions[maxIdx], newBoards[maxIdx], maxScore
	}

	// Calculate probability for each action.
	probabilities := make([]float64, len(scores))
	for ii, score := range scores {
		probabilities[ii] = score / rs.randomness
	}
	probabilities = softmax(probabilities)

	// Select from probabilities.
	chance := rand.Float64()
	log.Printf("chance=%f, scores=%v, probabilities=%v", chance, scores, probabilities)
	for ii, value := range probabilities {
		if chance <= value {
			return actions[ii], newBoards[ii], scores[ii]
		}
		chance -= value
	}
	log.Fatalf("Nothing selected!? final chance=%f", chance)
	return Action{}, nil, 0.0
}

func softmax(values []float64) (probs []float64) {
	probs = make([]float64, len(values))
	sum := 0.0
	// Normalize value for numeric values (smaller exponentials)
	for ii, value := range values {
		probs[ii] = math.Exp(value)
		sum += probs[ii]
	}
	for ii := range probs {
		probs[ii] /= sum
	}
	return
}

// NewRandomizedSearcher: take an action based on score associated to that action.
// Args:
//
//    searcher: Searcher to use after the first move.
//    randomness: Set to 0 to always take the action that maximizes the expected value (no
//      exploration). Otherwise works as divisor for the scores: larger values means more
//      randomness (exploration), smaller values means less randomness (exploitation).
func NewRandomizedSearcher(searcher Searcher, randomness float64) Searcher {
	return &randomizedSearcher{searcher: searcher, randomness: randomness}
}
