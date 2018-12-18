package search

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf
var _ = fmt.Printf

var flag_useActionProb = flag.Bool("ab_use_actions", false,
	"Use action probabilities to sort actions. Requires TensorFlow model.")
var flag_maxMoveRandomness = flag.Int("ab_max_move_randomness", 10+3,
	"After this move randomness is dropped and the game follows on without it.")

func printBoard(b *Board) {
	ui := ascii_ui.NewUI(true, false)
	ui.PrintBoard(b)
}

type abStats struct {
	nodes, evals, leafEvals int
	leafEvalsConsidered     int
	prunes                  int
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
//    randomness: If > 0, a random value of this magnitude is added to the leaf node values.
//
// Returns:
//    bestAction: that it suggests taking.
//    bestBoard: Board after taking bestAction.
//    bestScore: score of taking betAction
func AlphaBeta(board *Board, scorer ai.BatchScorer, maxDepth int, parallelize bool, randomness float32, stats *abStats) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	alpha := float32(-math.MaxFloat32)
	beta := float32(-math.MaxFloat32)
	if parallelize {
		// TODO: move to a parallelized version. Careful with stats, likely will need a mutex.
		bestAction, bestBoard, bestScore = alphaBetaRecursive(board, scorer, maxDepth, alpha, beta, randomness, stats)
	} else {
		bestAction, bestBoard, bestScore = alphaBetaRecursive(board, scorer, maxDepth, alpha, beta, randomness, stats)
	}
	return
}

var muLogBoard sync.Mutex

func TimedAlphaBeta(board *Board, scorer ai.BatchScorer, maxDepth int, parallelize bool, randomness float32) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	stats := abStats{}
	start := time.Now()
	bestAction, bestBoard, bestScore = AlphaBeta(board, scorer, maxDepth, parallelize, randomness, &stats)
	elapsedTime := time.Since(start).Seconds()
	// Yes, crazy right ? But without this the if below doesn't work ? TODO: report this, as of Go 1.11.2.
	// https://groups.google.com/forum/#!topic/golang-nuts/Yg09oBiRWbo
	// Looking at the assembly code generated, it seems related inlined time.go:790 code, that gets inserted
	// between the call to glog.V(2) and actually fetching its result ???
	hmm := bool(glog.V(3))
	_ = &hmm
	if hmm {
		muLogBoard.Lock()
		defer muLogBoard.Unlock()

		ui := ascii_ui.NewUI(true, false)
		fmt.Println()
		ui.PrintPlayer(board)
		fmt.Printf(" - Move #%d\n\n", board.MoveNumber)
		ui.PrintBoard(board)
		fmt.Println()
		var bestActionProb float32
		scores, vecActionsProbs := scorer.BatchScore([]*Board{board}, false)
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
		fmt.Printf("Best action found: %s - score=%.2f, αβ-score=%.2f, prob=%.2f%%\n\n",
			bestAction, scores[0], bestScore, bestActionProb*100)
	}
	if glog.V(2) {
		glog.V(2).Infof("Counts: %v", stats)
		evals := float64(stats.evals)
		leafEvals := float64(stats.leafEvals)
		leafEvalsConsidered := float64(stats.leafEvals)
		glog.V(2).Infof("  nodes/s=%.1f, evals/s=%.1f", float64(stats.nodes)/elapsedTime, evals/elapsedTime)
		glog.V(2).Infof("  leafEvals=%.2f%%, leafEvalsConsidered=%.2f%%", 100*leafEvals/evals, 100*leafEvalsConsidered/leafEvals)
	}
	return
}

func alphaBetaRecursive(board *Board, scorer ai.BatchScorer, maxDepth int, alpha, beta float32, randomness float32, stats *abStats) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	stats.nodes++

	// Sub-actions and boards available at this state.
	var actions []Action
	var newBoards []*Board
	var scores []float32
	if maxDepth > 1 || !*flag_useActionProb {
		actions, newBoards, scores = ExecuteAndScoreActions(board, scorer)
		stats.evals += len(actions)
		if maxDepth == 1 {
			stats.leafEvals += len(actions)
		}
	} else {
		// TODO: Instead of expanding the board and using V(s_{t+1}), take Q(s_t, a_t) instead, since it's cheaper.
		actions = board.Derived.Actions
		log.Panic("Q(s,a) learning not implemented yet.")
	}

	// If there are no valid actions, create the "pass" action
	if len(actions) == 1 && newBoards[0].IsFinished() {
		return actions[0], newBoards[0], scores[0]
	}
	if maxDepth <= 1 && randomness > 0 && board.MoveNumber <= *flag_maxMoveRandomness {
		for ii := range scores {
			scores[ii] += (rand.Float32()*2 - 1) * randomness
		}
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
			_, _, score := alphaBetaRecursive(newBoards[ii], scorer, maxDepth-1, beta, bestScore, randomness, stats)
			scores[ii] = -score
		}

		// Update best score.
		if scores[ii] > bestScore {
			bestScore = scores[ii]
			bestAction = actions[ii]
			bestBoard = newBoards[ii]
			if maxDepth == 1 {
				stats.leafEvalsConsidered++
			}
		}

		// Prune.
		if bestScore >= -beta {
			// The opponent will never take this path, so we can prune it.
			stats.prunes++
			return
		}
	}

	return
}

type alphaBetaSearcher struct {
	maxDepth     int
	parallelized bool
	randomness   float32

	scorer ai.BatchScorer
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

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPruning.
func NewAlphaBetaSearcher(maxDepth int, parallelized bool, scorer ai.BatchScorer, randomness float32) Searcher {
	return &alphaBetaSearcher{maxDepth: maxDepth, parallelized: parallelized, scorer: scorer, randomness: randomness}
}

// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions)+1.
func (ab *alphaBetaSearcher) ScoreMatch(b *Board, actions []Action, want []*Board) (
	scores []float32, actionsLabels [][]float32) {
	scores = make([]float32, 0, len(actions)+1)
	actionsLabels = make([][]float32, 0, len(actions))
	for actionIdx, action := range actions {
		bestAction, newBoard, score := TimedAlphaBeta(b, ab.scorer, ab.maxDepth, ab.parallelized, ab.randomness)
		glog.V(1).Infof("Move #%d (%d left), Action taken: %s / Best action examined %s (score=%.4g)",
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
		glog.V(1).Infof("Move #%d (%d left), Action taken: %s / Best action examined %s",
			b.MoveNumber, len(actions)-actionIdx-1, action, bestAction)
		if action == bestAction {
			b = newBoard
		} else {
			// Match action was different than what it would have played.
			b = b.Act(action)
		}
	}

	// Add the final board score, if the match hasn't ended yet.
	if isEnd, score := ai.EndGameScore(b); isEnd {
		scores = append(scores, score)
	} else {
		_, _, score = TimedAlphaBeta(b, ab.scorer, ab.maxDepth, ab.parallelized, ab.randomness)
		scores = append(scores, score)
	}
	return
}
