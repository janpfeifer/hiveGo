package search

import (
	"fmt"
	"log"
	"math"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf
var _ = fmt.Printf

type alphaBetaSearcher struct {
	maxDepth int
}

// Search implements the Searcher interface.
func (ab *alphaBetaSearcher) Search(b *Board, scorer ai.BatchScorer) (
	action Action, board *Board, score float32) {
	return AlphaBeta(b, scorer, ab.maxDepth)
}

func (ab *alphaBetaSearcher) ScoreMatch(b *Board, scorer ai.BatchScorer, actions []Action, reuse bool) (scores []float32) {
	log.Panicf("ScoreMatch not implemented for AlphaBetaSearcher")
	return
}

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPrunning.
func NewAlphaBetaSearcher(maxDepth int) Searcher {
	return &alphaBetaSearcher{maxDepth: maxDepth}
}

// Alpha Beta Pruning algorithm
// See: wikipedia.org/wiki/Alpha-beta_pruning
//
// TODO: Iterative deepening, princiapal variation estimation of scores.
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
	bestAction Action, bestBoard *Board, bestScore float32) {
	alpha := float32(-math.MaxFloat32)
	beta := float32(-math.MaxFloat32)
	bestAction, bestBoard, bestScore = alphaBetaRecursive(board, scorer, maxDepth, alpha, beta)
	glog.V(1).Infof("Estimated best score: %.2f", bestScore)
	return
}

func alphaBetaRecursive(board *Board, scorer ai.BatchScorer, maxDepth int, alpha, beta float32) (
	bestAction Action, bestBoard *Board, bestScore float32) {

	// If there are no valid actions, create the "pass" action
	actions, newBoards, scores := ScoredActions(board, scorer)
	if len(actions) == 1 && newBoards[0].IsFinished() {
		return actions[0], newBoards[0], scores[0]
	}
	SortActionsBoardsScores(actions, newBoards, scores)

	// The score to beat is the current "alpha" (best live score for current player)
	bestScore = alpha
	bestBoard = nil
	bestAction = Action{}

	// TODO: Sort actions according the expected score.
	for ii := range actions {
		if maxDepth > 1 && !newBoards[ii].IsFinished() {
			// Runs alphaBeta for opponent player, so the alpha/beta are reversed.
			_, _, score := alphaBetaRecursive(newBoards[ii], scorer, maxDepth-1, beta, bestScore)
			scores[ii] = -score
		}

		// Update best score.
		if scores[ii] > bestScore {
			bestScore = scores[ii]
			bestAction = actions[ii]
			bestBoard = newBoards[ii]
		}

		// Prune.
		if bestScore >= -beta {
			// The opponent will never take this path, so we can prune it.
			return
		}
	}

	return
}
