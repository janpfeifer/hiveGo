package alphabeta

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/janpfeifer/hiveGo/internal/searchers"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
)

// Searcher implements the searchers.Searcher interface.
// It is used by players.SearcherScorer, along with the scorer, to implement an AI player (players.Player interface).
type Searcher struct {
	maxDepth          int
	maxTime           time.Duration
	randomness        float32
	maxMoveRandomness int
	scorer            ai.BatchBoardScorer
	stats             Stats
}

// Assert that Searcher implements searchers.Searcher.
var _ searchers.Searcher = (*Searcher)(nil)

// Stats stores running stats collected during the search: for benchmarking, monitoring and debugging purposes.
type Stats struct {
	// nodes "played" during search -- execution of an action in a board, following by the creation of the new board.
	nodes int

	// evals means the number of examples passed to the scorer. Notice end-game situations are not scored and don't
	// count here.
	evals int

	leafEvals           int
	leafEvalsConsidered int
	prunes              int
}

// New returns an Alpha-Beta Pruning based searchers.Searcher implementation.
// There are many other optional configurations, see methods Searcher.With...
//
// The one obligatory parameter is the scorer used for the search.
//
// See: wikipedia.org/wiki/Alpha-beta_pruning
func New(scorer ai.BatchBoardScorer) *Searcher {
	return &Searcher{
		scorer:   scorer,
		maxDepth: DefaultMaxDepth,
	}
}

// DefaultMaxDepth for search.
const DefaultMaxDepth = 3

// WithMaxDepth sets a default max depth of search: the unit here are plies (ply singular). Each player
// playing counts as one ply. See https://en.wikipedia.org/wiki/Ply_(game_theory).
//
// This overrides WithMaxTime.
//
// The default is 3 (DefaultMaxDepth).
func (ab *Searcher) WithMaxDepth(maxDepth int) *Searcher {
	ab.maxDepth = maxDepth
	if maxDepth > 0 {
		ab.maxTime = 0
	} else {
		ab.maxDepth = 0
		// If disabling maxDepth, set maxTime to some default, if it is not set.
		if ab.maxTime == 0 {
			ab.maxTime = 3 * time.Second
		}
	}
	return ab
}

// WithRandomness adds a gaussian noise scaled to randomness to the scores returned by the scorer.
// Scores vary from -1 to 1 (+/- state.WinGameScore), so a value of 1.0 here would be a lot.
//
// This can be useful to make the AI play worse, to make it more fun.
//
// If noise is added, the scores are also further squashed by an S curve.
//
// Set to 0 to disable randomness -- this is the default.
//
// See also WithMaxMoveRandomness.
func (ab *Searcher) WithRandomness(randomness float32) *Searcher {
	ab.randomness = randomness
	return ab
}

// WithMaxMoveRandomness sets a move limit after which randomness is disabled.
//
// This is desirable if, for instance, using randomness only to generate different openings.
func (ab *Searcher) WithMaxMoveRandomness(maxMoveRandomness int) *Searcher {
	ab.maxMoveRandomness = maxMoveRandomness
	return ab
}

// WithMaxTime sets a default max duration of thinking per search.
// This overrides WithMaxDepth.
//
// The default is no time-limit, and instead be limited by WithMaxDepth.
func (ab *Searcher) WithMaxTime(maxTime time.Duration) *Searcher {
	ab.maxTime = maxTime
	if maxTime > 0 {
		ab.maxDepth = 0
	} else {
		ab.maxTime = 0
		// If disabling maxTime, set maxDepth to default, if it is not set.
		if ab.maxDepth == 0 {
			ab.maxDepth = 2
		}
	}
	return ab
}

// Search implements the Searcher interface.
//
// It returns actionLabels always nil, because it wouldn't be a good approximation for the non-best move.
// This is because of the pruning aspect of the algorithm: bad moves are cut short, so alpha-beta pruning score
// estimation for bad moves will not be a good one.
func (ab *Searcher) Search(b *Board) (action Action, board *Board, score float32, actionsLabels []float32) {
	start := time.Now()
	actionsLabels = nil
	// TODO: implement maxTime by interactively increasing the depth in the search, until the time expires.
	action, board, score = ab.searchToMaxDepth(board, ab.maxDepth)
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
		batchScores := ab.scorer.BatchBoardScore([]*Board{board})
		fmt.Printf("Best action found: %s - shallow score=%.2f, αβ-score=%.2f\n\n",
			action, batchScores[0], score)
	}
	if klog.V(2).Enabled() {
		klog.Infof("Counts: %+v", ab.stats)
		evals := float64(ab.stats.evals)
		leafEvals := float64(ab.stats.leafEvals)
		leafEvalsConsidered := float64(ab.stats.leafEvals)
		klog.Infof("  nodes/s=%.1f, evals/s=%.1f", float64(ab.stats.nodes)/elapsedTime, evals/elapsedTime)
		klog.Infof("  leafEvals=%.2f%%, leafEvalsConsidered=%.2f%%", 100*leafEvals/evals, 100*leafEvalsConsidered/leafEvals)
	}
	return
}

// searchToMaxDepth executes alpha-beta pruning algorithm to the given depth.
// Returns:
//
//	bestAction: that it suggests taking.
//	bestBoard: Board after taking bestAction.
//	bestScore: score of taking betAction
//
// TODO: Add support to a parallelized version. Careful with stats, likely will need a mutex.
func (ab *Searcher) searchToMaxDepth(board *Board, maxDepth int) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	alpha := float32(-math.MaxFloat32)
	beta := float32(-math.MaxFloat32)
	addNoise := ab.randomness > 0 && (ab.maxMoveRandomness <= 0 || board.MoveNumber <= ab.maxMoveRandomness)
	bestAction, bestBoard, bestScore = ab.recursion(board, maxDepth, alpha, beta, addNoise)
	return
}

var muLogBoard sync.Mutex

// recursion of the alpha-beta pruning algorithm, with depthLeft plies to go.
func (ab *Searcher) recursion(board *Board, depthLeft int, alpha, beta float32, addNoise bool) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	isLeaf := depthLeft <= 1

	// Sub-actions and boards available at this state: in principle we would only need to score the leaf
	// nodes, but we score intermediary nodes to guide the alpha-beta pruning search -- it prunes more
	// if we search for the better nodes first and find high values for alpha, making it faster overall.
	actions := board.Derived.Actions
	newBoards, scores := executeAndScoreActions(board, ab.scorer)
	ab.stats.nodes += len(newBoards)

	// If there is only one action, and it leads to and end-game, then there is nothing else to explore.
	if len(actions) == 1 && newBoards[0].IsFinished() {
		return actions[0], newBoards[0], scores[0]
	}

	// If there is a winning move, the scorer was not used (no evals), and we take the winning move (or one of them at random),
	// no need to explore deeper.
	bestActionIdx := -1
	winningMoves := 0
	for actionIdx, score := range scores {
		if score == ai.WinGameScore {
			winningMoves++
			if winningMoves == 1 || rand.Intn(winningMoves) == 0 {
				bestActionIdx = actionIdx
			}
		}
	}
	if winningMoves > 0 {
		return actions[bestActionIdx], newBoards[bestActionIdx], ai.WinGameScore
	}

	// Count actual evals.
	ab.stats.evals += len(scores)
	if isLeaf {
		ab.stats.leafEvals += len(scores)
	}

	// Leaf nodes take the score returned by the scorer.
	if isLeaf {
		// Add noise to leaf nodes if randomness was configured:
		if addNoise {
			// Randomize only non end-of-game actions
			for ii := range scores {
				if !newBoards[ii].IsFinished() {
					noise := float32(rand.NormFloat64()*float64(ab.randomness)) * ai.WinGameScore
					scores[ii] += ai.SquashScore(scores[ii] + noise)
				}
			}
		}
	}

	// Find order from the best scoring first.
	bestScore = float32(-math.MaxFloat32)
	bestBoard = nil
	bestAction = Action{}
	ordering := generics.SliceOrdering(scores, true) // Reverse order by score.
	for _, actionIdx := range ordering {
		if searchers.IdleChan != nil {
			// Wait for an "idle" signal before each search: only used in WASM.
			<-searchers.IdleChan
		}

		// Only follows recursion if this action doesn't end the match.
		if !newBoards[actionIdx].IsFinished() {
			// Runs alphaBeta for opponent player, so the alpha/beta are reversed.
			_, _, score := ab.recursion(newBoards[actionIdx], depthLeft-1, beta, alpha, addNoise)
			// the score is the negative of the opponents score.
			scores[actionIdx] = -score
		}

		// Update the alpha for pruning.
		if scores[actionIdx] > alpha {
			alpha = scores[actionIdx]
		}

		// Save bestScore for this board.
		if scores[actionIdx] > bestScore {
			bestScore = scores[actionIdx]
			bestAction = actions[actionIdx]
			bestBoard = newBoards[actionIdx]
		}

		// Prune.
		if -bestScore <= beta {
			// The opponent will never take this path, so we can prune the search and stop here.
			ab.stats.prunes++
			return
		}

		// If bestScore is a win, it can stop early.
		if bestBoard != nil && bestBoard.IsFinished() && bestScore > 0 {
			// This is a winner move, no need to look further.
			return
		}
	}

	return bestAction, bestBoard, bestScore
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
