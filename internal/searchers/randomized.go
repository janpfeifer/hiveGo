package searchers

import (
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/hiveGo/internal/ai"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"log"
	"math"
	"math/rand/v2"
	"slices"
)

// randomizedSearcher is a meta Searcher, that introduces randomness to its scorer.
type randomizedSearcher struct {
	searcher   Searcher
	scorer     ai.BatchBoardScorer
	randomness float64
}

// Search implements the Searcher interface.
func (rs *randomizedSearcher) Search(b *Board) (Action, *Board, float32, []float32) {
	// If there are no valid actions, create the "pass" action
	actions, newBoards, scores := ExecuteAndScoreActions(b, rs.scorer)

	for ii := range actions {
		isEnded, score := ai.EndGameScore(newBoards[ii])
		if !isEnded {
			_, _, scores[ii], _ = rs.searcher.Search(newBoards[ii])
			scores[ii] = -scores[ii]
		} else {
			if !newBoards[ii].Draw() && newBoards[ii].Winner() == b.NextPlayer {
				return actions[ii], newBoards[ii], -score, ai.OneHotEncoding(len(actions), ii)
			}
			scores[ii] = -score
		}
	}

	// Calculate probability for each action.
	probabilities := make([]float64, len(scores))
	for ii, score := range scores {
		probabilities[ii] = float64(score) / rs.randomness
	}
	probabilities = softmax(probabilities)
	actionsLabels := make([]float32, len(probabilities))
	for ii, prob := range probabilities {
		actionsLabels[ii] = float32(prob)
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
		klog.V(1).Infof("Estimated best score: %.2f", maxScore)
		return actions[maxIdx], newBoards[maxIdx], maxScore, actionsLabels
	}

	// Select from probabilities.
	chance := rand.Float64()
	// klog.Infof("chance=%f, scores=%v, probabilities=%v", chance, scores, probabilities)
	for ii, value := range probabilities {
		if chance <= value {
			klog.V(1).Infof("Score of selected action (%s): %.2f", actions[ii], scores[ii])
			return actions[ii], newBoards[ii], scores[ii], actionsLabels
		}
		chance -= value
	}
	exceptions.Panicf("Nothing selected!? final chance=%f", chance)
	return Action{}, nil, 0.0, nil
}

func (rs *randomizedSearcher) ScoreMatch(b *Board, actions []Action) (
	scores []float32, actionsLabels [][]float32) {
	log.Panicf("ScoreMatch not implemented for RandomizedSearcher")
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

// NewRandomizedSearcher takes an action based on score associated to that action.
// Args:
//
//	searcher: Searcher to use after the first move.
//	randomness: Set to 0 to always take the action that maximizes the expected value (no
//	  exploration). Otherwise works as divisor for the scores: larger values means more
//	  randomness (exploration), smaller values means less randomness (exploitation).
func NewRandomizedSearcher(searcher Searcher, scorer ai.BatchBoardScorer, randomness float64) Searcher {
	return &randomizedSearcher{searcher: searcher, scorer: scorer, randomness: randomness}
}
