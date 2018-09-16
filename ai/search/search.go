package search

import (
	"log"
	"math"
	"math/rand"
	"sort"

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

	// ScoreMatch will score the board at each board position, starting from the current one,
	// and following each one of the actions. In the end, len(scores) == len(actions)+1.
	ScoreMatch(b *Board, scorer ai.BatchScorer, actions []Action) (scores []float64)
}

type randomizedSearcher struct {
	searcher   Searcher
	randomness float64
}

// ScoredActions enumerates each of the available actions, along with the boards
// where actions were taken and with the score for current b.NextPlayer -- not the
// next action's NextPlayer. It wil return early if any of the actions lead to
// b.NextPlayer winning.
func ScoredActions(b *Board, scorer ai.BatchScorer) ([]Action, []*Board, []float64) {
	actions := b.Derived.Actions
	if len(actions) == 0 {
		actions = append(actions, Action{Piece: NO_PIECE})
	}
	scores := make([]float64, len(actions))
	newBoards := make([]*Board, len(actions))

	// Pre-score actions that lead to end-game.
	boardsToScore := make([]*Board, 0, len(actions))
	hasWinning := 0
	for ii, action := range actions {
		newBoards[ii] = b.Act(action)
		if isEnd, score := ai.EndGameScore(newBoards[ii]); isEnd {
			// End game is treated differently.
			score = -score // Score for b.NextPlayer, not newBoards[ii].NextPlayer
			if score > 0.0 {
				hasWinning++
			}
			scores[ii] = score
		} else {
			boardsToScore = append(boardsToScore, newBoards[ii])
		}
	}

	// Player wins, return only the winning actions.
	if hasWinning > 0 {
		revisedActions := make([]Action, 0, hasWinning)
		revisedBoards := make([]*Board, 0, hasWinning)
		revisedScores := make([]float64, 0, hasWinning)
		for ii, action := range actions {
			if newBoards[ii].IsFinished() && scores[ii] > 0 {
				revisedActions = append(revisedActions, action)
				revisedBoards = append(revisedBoards, newBoards[ii])
				revisedScores = append(revisedScores, scores[ii])
			}
		}
		return revisedActions, revisedBoards, revisedScores
	}

	if len(boardsToScore) > 0 {
		// Score other boards.
		// TODO: Use "Principal Variation" to estimate the score.
		scored := scorer.BatchScore(boardsToScore)
		scoredIdx := 0
		for ii := range scores {
			if !newBoards[ii].IsFinished() {
				scores[ii] = -scored[scoredIdx] // Score for b.NextPlayer, not newBoards[ii].NextPlayer
				scoredIdx++
			}
		}
	}

	return actions, newBoards, scores
}

func SortActionsBoardsScores(actions []Action, boards []*Board, scores []float64) {
	s := &ScoresToSort{actions, boards, scores}
	sort.Sort(s)
}

// ScoreToSort provide a way to sort actions/boards/scores.
type ScoresToSort struct {
	actions []Action
	boards  []*Board
	scores  []float64
}

func (s *ScoresToSort) Swap(i, j int) {
	s.actions[i], s.actions[j] = s.actions[j], s.actions[i]
	s.boards[i], s.boards[j] = s.boards[j], s.boards[i]
	s.scores[i], s.scores[j] = s.scores[j], s.scores[i]
}
func (s *ScoresToSort) Len() int           { return len(s.scores) }
func (s *ScoresToSort) Less(i, j int) bool { return s.scores[i] > s.scores[j] }

// Search implements the Searcher interface.
func (rs *randomizedSearcher) Search(b *Board, scorer ai.BatchScorer) (Action, *Board, float64) {
	// If there are no valid actions, create the "pass" action
	actions, newBoards, scores := ScoredActions(b, scorer)

	for ii := range actions {
		if !newBoards[ii].IsFinished() {
			_, _, scores[ii] = rs.searcher.Search(newBoards[ii], scorer)
			scores[ii] = -scores[ii]
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
	// log.Printf("chance=%f, scores=%v, probabilities=%v", chance, scores, probabilities)
	for ii, value := range probabilities {
		if chance <= value {
			return actions[ii], newBoards[ii], scores[ii]
		}
		chance -= value
	}
	log.Fatalf("Nothing selected!? final chance=%f", chance)
	return Action{}, nil, 0.0
}

func (rs *randomizedSearcher) ScoreMatch(b *Board, scorer ai.BatchScorer, actions []Action) (scores []float64) {
	log.Panicf("ScoreMatch not implemented for RandomizedSearcher")
	return
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
