package ab

import (
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/features"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/janpfeifer/hiveGo/internal/searchers"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
)

type alphaBetaSearcher struct {
	maxDepth     int
	parallelized bool
	randomness   float32
	scorer       ai.BatchBoardScorer

	// Player parameter that indicates that Alpha-Beta-Pruning was selected.
	useAB bool
}

type abStats struct {
	nodes, evals, leafEvals int
	leafEvalsConsidered     int
	prunes                  int
}

func printBoard(b *Board) {
	ui := cli.New(true, false)
	ui.PrintBoard(b)
}

// AlphaBeta Pruning algorithm
// See: wikipedia.org/wiki/Alpha-beta_pruning
//
// TODO: Iterative deepening, principal variation estimation of scores.
//
// Args:
//
//	board: current board
//	scorer: batch scores boards.
//	maxDepth: How deep to make the searchers.
//	parallelize: Parallelize searchers, only first depth is parallelized.
//	randomness: If > 0, a random value of this magnitude is added to the leaf node values.
//
// Returns:
//
//	bestAction: that it suggests taking.
//	bestBoard: Board after taking bestAction.
//	bestScore: score of taking betAction
//
// TODO: Add support to a parallelized version. Careful with stats, likely will need a mutex.
func AlphaBeta(board *Board, scorer ai.BatchBoardScorer, maxDepth int, parallelize bool, randomness float32, stats *abStats) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	alpha := float32(-math.MaxFloat32)
	beta := float32(-math.MaxFloat32)
	bestAction, bestBoard, bestScore = alphaBetaRecursive(
		board, scorer, maxDepth, alpha, beta, randomness, stats)
	return
}

var muLogBoard sync.Mutex

func TimedAlphaBeta(board *Board, scorer ai.BatchBoardScorer, maxDepth int, parallelize bool, randomness float32) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	stats := abStats{}
	start := time.Now()
	bestAction, bestBoard, bestScore = AlphaBeta(board, scorer, maxDepth, parallelize, randomness, &stats)
	elapsedTime := time.Since(start).Seconds()
	if klog.V(3).Enabled() {
		muLogBoard.Lock()
		defer muLogBoard.Unlock()

		ui := cli.New(true, false)
		fmt.Println()
		ui.PrintPlayer(board)
		fmt.Printf(" - Move #%d\n\n", board.MoveNumber)
		ui.PrintBoard(board)
		fmt.Println()
		var bestActionProb float32
		scores := scorer.BatchBoardScore([]*Board{board})

		/*
			if vecActionsProbs != nil {
				actionsProbs := vecActionsProbs[0]
				if len(actionsProbs) > 0 {
					fmt.Printf("\nTop moves probs:\n")
					bestActionProb = actionsProbs[board.FindActionDeep(bestAction)]
					indices := make([]int, len(actionsProbs))
					for ii := range indices {
						indices[ii] = ii
					}
					sort.Slice(indices, func(i, j int) bool { return actionsProbs[indices[i]] > actionsProbs[indices[j]] })
					for ii := 0; ii < 5 && ii < len(indices); ii++ {
						action := board.Derived.Actions[indices[ii]]
						prob := actionsProbs[indices[ii]]
						if prob < 0.02 {
							break
						}
						fmt.Printf("\t%s - %.2f%%\n", action, prob*100)
					}
				}
			}
		*/
		fmt.Printf("Best action found: %s - score=%.2f, αβ-score=%.2f, prob=%.2f%%\n\n",
			bestAction, scores[0], bestScore, bestActionProb*100)
	}
	if klog.V(2).Enabled() {
		klog.Infof("Counts: %v", stats)
		evals := float64(stats.evals)
		leafEvals := float64(stats.leafEvals)
		leafEvalsConsidered := float64(stats.leafEvals)
		klog.Infof("  nodes/s=%.1f, evals/s=%.1f", float64(stats.nodes)/elapsedTime, evals/elapsedTime)
		klog.Infof("  leafEvals=%.2f%%, leafEvalsConsidered=%.2f%%", 100*leafEvals/evals, 100*leafEvalsConsidered/leafEvals)
	}
	return
}

func alphaBetaRecursive(board *Board, scorer ai.BatchBoardScorer, maxDepth int, alpha, beta float32,
	randomness float32, stats *abStats) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	stats.nodes++

	// Sub-actions and boards available at this state.
	actions := board.Derived.Actions
	newBoards, scores := executeAndScoreActions(board, scorer)
	stats.evals += len(scores)
	if maxDepth == 1 {
		stats.leafEvals += len(scores)
	}
	if len(actions) == 1 && newBoards[0].IsFinished() {
		return actions[0], newBoards[0], scores[0]
	}

	if maxDepth <= 1 && randomness > 0 && board.MoveNumber <= *flag_maxMoveRandomness {
		// Randomize only non end-of-game actions
		for ii := range scores {
			if !newBoards[ii].IsFinished() {
				if randomness > 0 {
					noise := float32(rand.NormFloat64()*float64(randomness)) * ai.WinGameScore
					scores[ii] += ai.SquashScore(scores[ii] + noise)
				}
			}
		}
	}
	sortActionsBoardsAndScores(actions, newBoards, scores)

	// The score to beat is the current "alpha" (best live score for current player)
	bestScore = alpha
	bestBoard = nil
	bestAction = Action{}
	for ii := range actions {
		if searchers.IdleChan != nil {
			// Wait for an "idle" signal before each search.
			<-searchers.IdleChan
		}
		if maxDepth > 1 && !newBoards[ii].IsFinished() {
			// Runs alphaBeta for opponent player, so the alpha/beta are reversed.
			_, _, score := alphaBetaRecursive(newBoards[ii], scorer, maxDepth-1, beta, bestScore, randomness, stats)
			scores[ii] = -score
		}

		// addResult best score.
		if scores[ii] > bestScore {
			bestScore = scores[ii]
			bestAction = actions[ii]
			bestBoard = newBoards[ii]
			if maxDepth == 1 {
				stats.leafEvalsConsidered++
			}
		}

		// Prune.
		if -bestScore <= beta {
			// The opponent will never take this path, so we can prune it.
			stats.prunes++
			return
		}

		// If bestScore is a win, it can stop early.
		if bestBoard != nil && bestBoard.IsFinished() && bestScore > 0 {
			// This is a winner move, no need to look further.
			return
		}
	}

	return
}

// Search implements the Searcher interface.
func (ab *alphaBetaSearcher) Search(b *Board) (action Action, board *Board, score float32, actionsLabels []float32) {
	action, board, score = TimedAlphaBeta(b, ab.scorer, ab.maxDepth, ab.parallelized, ab.randomness)
	actionsLabels = make([]float32, len(b.Derived.Actions))
	if !action.IsSkipAction() {
		actionsLabels[b.FindAction(action)] = 1
	}
	return
}

// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions).
func (ab *alphaBetaSearcher) ScoreMatch(b *Board, actions []Action) (
	scores []float32, actionsLabels [][]float32) {
	scores = make([]float32, 0, len(actions)+1)
	actionsLabels = make([][]float32, 0, len(actions))
	for actionIdx, action := range actions {
		bestAction, newBoard, score := TimedAlphaBeta(b, ab.scorer, ab.maxDepth, ab.parallelized, ab.randomness)
		klog.V(1).Infof("Move #%d (%d left), Action taken: %s / Best action examined %s (score=%.4g)",
			b.MoveNumber, len(actions)-actionIdx-1, action, bestAction, score)
		scores = append(scores, score)
		if len(b.Derived.Actions) > 1 {
			// AlphaBetaPruning policy is binary, effectively being one-hot-encoding.
			bestActionIdx := b.FindAction(bestAction)
			bestActionVec := make([]float32, len(b.Derived.Actions))
			bestActionVec[bestActionIdx] = 1
			actionsLabels = append(actionsLabels, bestActionVec)
		} else {
			actionsLabels = append(actionsLabels, nil)
		}
		if action == bestAction {
			b = newBoard
		} else {
			// Match action was different than what it would have played.
			b = b.Act(action)
		}
	}
	return
}

// executeAndScoreActions creates the boards after executing each of the board actions,
// and returns the new boards and their scores according to the given scorer.
//
// It returns without using the scorer if any of the actions lead to b.NextPlayer winning.
func executeAndScoreActions(board *Board, scorer ai.BatchBoardScorer) (newBoards []*Board, scores []float32) {
	actions := board.Derived.Actions
	scores = make([]float32, len(actions))
	newBoards = make([]*Board, len(actions))

	// Pre-score actions that lead to end-game.
	boardsToScore := make([]*Board, 0, len(actions))
	hasWinning := 0
	for ii, action := range actions {
		newBoards[ii] = board.Act(action)
		if isEnd, score := ai.IsEndGameAndScore(newBoards[ii]); isEnd {
			// End game is treated differently.
			score = -score // Score for board.NextPlayer, not newBoards[ii].NextPlayer
			if score > 0.0 {
				hasWinning++
			}
			scores[ii] = score
		} else {
			boardsToScore = append(boardsToScore, newBoards[ii])
		}
	}

	// Player wins, no need to score the other actions.
	if hasWinning > 0 {
		// Actually we could just trim return only the winning action(s). But
		// for ML training, it's useful to return all and let it learn from it.
		// But in any cases there is no need to rescore the other
		// actions.
		return
	}

	if len(boardsToScore) > 0 {
		// Score non-game ending boards.
		// TODO: Use "Principal Variation" to estimate the score.
		scored := scorer.BatchBoardScore(boardsToScore)
		scoredIdx := 0
		for ii := range scores {
			if !newBoards[ii].IsFinished() {
				// Score for board.NextPlayer, not newBoards[ii].NextPlayer, hence
				// we take the inverse here.
				scores[ii] = -scored[scoredIdx]
				scoredIdx++
			}
		}
	}
	return
}

// sortActionsBoardsAndScores jointly based on scores: higher scores first.
func sortActionsBoardsAndScores(actions []Action, boards []*Board, scores []float32) {
	s := &scoresToSort{actions, boards, scores}
	sort.Sort(s)
}

// scoresToSort provides a way to jointly sort actions, boards and scores by their scores -- larger scores come first.
type scoresToSort struct {
	actions []Action
	boards  []*Board
	scores  []float32
}

func (s *scoresToSort) Swap(i, j int) {
	s.actions[i], s.actions[j] = s.actions[j], s.actions[i]
	s.boards[i], s.boards[j] = s.boards[j], s.boards[i]
	s.scores[i], s.scores[j] = s.scores[j], s.scores[i]
}
func (s *scoresToSort) Len() int           { return len(s.scores) }
func (s *scoresToSort) Less(i, j int) bool { return s.scores[i] > s.scores[j] }
