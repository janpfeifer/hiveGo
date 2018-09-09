package search

import (
	"log"
	"math"

	"github.com/janpfeifer/hiveGo/ai"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf

type alphaBetaSearcher struct {
	maxDepth int
}

// Search implements the Searcher interface.
func (ab *alphaBetaSearcher) Search(b *Board, scorer ai.BatchScorer) (
	action Action, board *Board, score float64) {
	return AlphaBeta(b, scorer, ab.maxDepth)
}

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPrunning.
func NewAlphaBetaSearcher(maxDepth int) Searcher {
	return &alphaBetaSearcher{maxDepth: maxDepth}
}

// Alpha Beta Pruning algorithm
// See: wikipedia.org/wiki/Alpha-beta_pruning
//
// Args:
//    board: current board
//    scorer: batch scores boards.
//    maxDepth: How deep to make the search.
//
// Returns:
//    bestAction: that it suggests taking.
//    bestBoard: Board after taking bestAction.
//    bestScore: score of taking betAction
func AlphaBeta(board *Board, scorer ai.BatchScorer, maxDepth int) (
	bestAction Action, bestBoard *Board, bestScore float64) {
	alpha := -math.MaxFloat64
	beta := math.MaxFloat64
	return alphaBetaRecursive(board, scorer, maxDepth, alpha, beta)
}

func alphaBetaRecursive(board *Board, scorer ai.BatchScorer, maxDepth int, alpha, beta float64) (
	bestAction Action, bestBoard *Board, bestScore float64) {

	// The score to beat is the current "alpha" (best live score for current player)
	bestScore = alpha
	bestBoard = nil
	bestAction = Action{}

	// If there are no valid actions, create the "pass" action
	actions := board.Derived.Actions
	if len(actions) == 0 {
		actions = append(actions, Action{Piece: NO_PIECE})
	}

	// TODO: Sort actions according the expected score.
	for _, action := range actions {
		// Execute action and evaluate expected score of taking it.
		newB := board.Act(action)

		// Use standard end game scores if game is finished.
		isEnd, score := ai.EndGameScore(newB)
		if !isEnd {
			if maxDepth == 1 {
				score = scorer.Score(newB)
			} else {
				// Runs alphaBeta for opponent player, so the alpha/beta are reversed.
				_, _, score = alphaBetaRecursive(newB, scorer, maxDepth-1, beta, bestScore)
			}
			// Score for player that started the move is the reverse of the score
			// of the opponent player (the one playing in newB)
			score = -score
		}

		// Update best score.
		if score > bestScore {
			bestScore = score
			bestAction = action
			bestBoard = newB
		}

		// Prune.
		if bestScore >= beta {
			// The opponent will never take this path, so we can prune it.
			return
		}
	}

	return
}
