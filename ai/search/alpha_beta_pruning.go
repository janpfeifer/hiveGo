package search

import (
	"fmt"
	"log"
	"math"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf
var _ = fmt.Printf

func printBoard(b *Board) {
	ui := ascii_ui.NewUI(true, false)
	ui.PrintBoard(b)
}

// Alpha Beta Pruning algorithm
// See: wikipedia.org/wiki/Alpha-beta_pruning
//
// TODO: Iterative deepening, principal variation estimation of scores.
//
// Args:
//    board: current board
//    scorer: batch scores boards.
//    maxDepth: How deep to make the search.
//    parallelize: Parallelize search, only first depth is parallelized.
//
// Returns:
//    bestAction: that it suggests taking.
//    bestBoard: Board after taking bestAction.
//    bestScore: score of taking betAction
func AlphaBeta(board *Board, scorer ai.BatchScorer, maxDepth int, parallelize bool) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	alpha := float32(-math.MaxFloat32)
	beta := float32(-math.MaxFloat32)
	if parallelize {
		// TODO: move to a parallelized version.
		bestAction, bestBoard, bestScore = alphaBetaRecursive(board, scorer, maxDepth, alpha, beta)
	} else {
		bestAction, bestBoard, bestScore = alphaBetaRecursive(board, scorer, maxDepth, alpha, beta)
	}
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
	for ii := range actions {
		if IdleChan != nil {
			// Wait for an "idle" signal before each search.
			<-IdleChan
		}
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

func alphaBetaParallelized(board *Board, scorer ai.BatchScorer, maxDepth int, alpha, beta float32) (
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

type alphaBetaSearcher struct {
	maxDepth     int
	parallelized bool

	scorer ai.BatchScorer
}

// Search implements the Searcher interface.
func (ab *alphaBetaSearcher) Search(b *Board) (action Action, board *Board, score float32, actionsLabels []float32) {
	action, board, score = AlphaBeta(b, ab.scorer, ab.maxDepth, ab.parallelized)
	actionsLabels = make([]float32, len(b.Derived.Actions))
	actionsLabels[b.FindAction(action)] = 1
	return
}

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPruning.
func NewAlphaBetaSearcher(maxDepth int, parallelized bool, scorer ai.BatchScorer) Searcher {
	return &alphaBetaSearcher{maxDepth: maxDepth, parallelized: parallelized, scorer: scorer}
}

// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions)+1.
func (ab *alphaBetaSearcher) ScoreMatch(b *Board, actions []Action) (
	scores []float32, actionsLabels [][]float32) {
	scores = make([]float32, 0, len(actions)+1)
	actionsLabels = make([][]float32, 0, len(actions))
	for _, action := range actions {
		bestAction, newBoard, score := AlphaBeta(b, ab.scorer, ab.maxDepth, ab.parallelized)
		scores = append(scores, score)
		if len(b.Derived.Actions) > 0 {
			// AlphaBetaPrunning policy is binary, effectively being one-hot-encoding.
			bestActionIdx := b.FindAction(bestAction)
			bestActionVec := make([]float32, len(b.Derived.Actions))
			bestActionVec[bestActionIdx] = 1
			actionsLabels = append(actionsLabels, bestActionVec)
		} else {
			actionsLabels = append(actionsLabels, nil)
		}
		glog.V(1).Infof("Move %d, Player %d, Score %.2f", b.MoveNumber, b.NextPlayer, score)
		if action == bestAction {
			glog.V(1).Infof("  Action: %s", action)
			b = newBoard
		} else {
			// Match action was different than what it would have played.
			glog.V(1).Infof("  AI would have played %s instead of %s", bestAction, action)
			b = b.Act(action)
		}
		if glog.V(2) {
			printBoard(b)
			//fmt.Printf("Actions: %v\n", b.Derived.Actions)
			fmt.Println("")
		}
	}

	// Add the final board score, if the match hasn't ended yet.
	if isEnd, score := ai.EndGameScore(b); isEnd {
		scores = append(scores, score)
	} else {
		_, _, score = AlphaBeta(b, ab.scorer, ab.maxDepth, ab.parallelized)
		scores = append(scores, score)
	}
	return
}
