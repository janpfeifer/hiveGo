package alphabeta

import (
	"context"
	"fmt"
	"github.com/chewxy/math32"
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/janpfeifer/hiveGo/internal/searchers"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
)

// Searcher implements the searchers.Searcher interface.
// It is used by players.SearcherScorer, along with the scorer, to implement an AI player (players.Player interface).
type Searcher struct {
	maxDepth          int
	discount          float32 // \lambda parameter in TD-lambda scoring: how future scores are discounted to current state.
	maxTime           time.Duration
	randomness        float32
	maxMoveRandomness int
	scorer            ai.BatchValueScorer
	stats             Stats

	drawScore float32
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
func New(scorer ai.ValueScorer) *Searcher {
	batchScorer, ok := scorer.(ai.BatchValueScorer)
	if !ok {
		batchScorer = &ai.BatchBoardScorerProxy{ValueScorer: scorer}
	}
	return &Searcher{
		scorer:            batchScorer,
		discount:          DefaultDiscount,
		maxMoveRandomness: 10,
	}
}

const (
	// DefaultDiscount for score of future states.
	DefaultDiscount = 0.98
)

// NewFromParams configures the alpha-beta pruning search with the parameters given if "ab" is set to true.
// Otherwise, it returns nil (and no error).
//
// It pops out the parameters used (see parameters.PopParamOr).
//
// Params used:
//
// - "max_depth": in number of plies to search. A good number is 3. Either max_depth or max_time must be set.
// - "max_time": it iteratively searches deeper while there is time. Either max_depth or max_time must be set.
// - "discount": multiplicative discount the score per ply. Default is 0.98.
// - "randomness": adds a normal noise to the scores with mean 0 and the given value as standard deviation. Default is 0.
// - "max_move_rand": max move (in plies) of the game to which to apply randomness. After this move, randomness is set to 0.
// - "draw_score": (-1 to +1) how much score to associate to a draw. If you want to skew the AI to avoid draws, set to some negative value.
func NewFromParams(scorer ai.ValueScorer, params parameters.Params) (searchers.Searcher, error) {
	isAB, err := parameters.PopParamOr(params, "ab", false)
	if err != nil {
		return nil, err
	}
	if !isAB {
		return nil, nil
	}
	ab := New(scorer)

	ab.maxDepth, err = parameters.PopParamOr(params, "max_depth", ab.maxDepth)
	if err != nil {
		return nil, err
	}
	ab.maxTime, err = parameters.PopParamOr(params, "max_time", ab.maxTime)
	if err != nil {
		return nil, err
	}
	ab.discount, err = parameters.PopParamOr(params, "discount", ab.discount)
	if err != nil {
		return nil, err
	}
	ab.randomness, err = parameters.PopParamOr(params, "randomness", ab.randomness)
	if err != nil {
		return nil, err
	}
	ab.maxMoveRandomness, err = parameters.PopParamOr(params, "max_move_rand", ab.maxMoveRandomness)
	if err != nil {
		return nil, err
	}
	ab.drawScore, err = parameters.PopParamOr(params, "draw_score", ab.drawScore)
	if err != nil {
		return nil, err
	}

	if ab.maxDepth == 0 && ab.maxTime == 0 {
		return nil, errors.Errorf("invalid \"ab\" configuration: either max_depth or max_time must be set (Alpha-Beta Pruning)")
	}
	return ab, nil
}

// String implements searchers.Searcher.
func (ab *Searcher) String() string {
	parts := []string{"αβ-prunning"}
	if ab.maxTime > 0 {
		parts = append(parts, fmt.Sprintf("max_time=%s", ab.maxTime))
	}
	if ab.maxDepth > 0 {
		parts = append(parts, fmt.Sprintf("max_depth=%s", ab.maxDepth))
	}
	if ab.randomness != 1 {
		parts = append(parts, fmt.Sprintf("randomness=%f", ab.randomness))
	}
	return strings.Join(parts, ", ")
}

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
// Scores vary from -1 to 1 (+/- ai.WinGameScore), so a value of 1.0 here would be a lot.
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

// WithDiscount sets a discount factor for future moves in the alpha-beta evaluation.
// A lower discount factor penalizes future moves more heavily, which could make the search favor
// moves that have immediate benefits.
//
// Valid values for discount are in the range [0.0, 1.0]. A value of 1.0 means no discount is applied,
// whereas smaller values disproportionately reduce the importance of deeper moves.
//
// This can be useful to model scenarios where long-term gains are less certain or desirable.
//
// Default is 1.0 (no discount applied).
func (ab *Searcher) WithDiscount(discount float32) *Searcher {
	if discount < 0.0 || discount > 1.0 {
		exceptions.Panicf("invalid parameter WithDiscount(%.4g), discount must be between 0.0 and 1.0", discount)
	}
	ab.discount = discount
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

// Search implements the searchers.Searcher interface.
//
// It returns actionLabels always nil, because it wouldn't be a good approximation for the non-best move.
// This is because of the pruning aspect of the algorithm: bad moves are cut short, so alpha-beta pruning score
// estimation for bad moves will not be a good one.
func (ab *Searcher) Search(board *Board) (bestAction Action, bestBoard *Board, bestScore float32, err error) {
	start := time.Now()
	if board.Derived == nil {
		board.BuildDerived()
	}

	if ab.maxTime == 0 {
		// Search to fixed maxDepth:
		bestAction, bestBoard, bestScore = ab.searchToMaxDepth(nil, board, ab.maxDepth, board.NextPlayer)
	} else {
		deadline := time.Now().Add(ab.maxTime)

		// First depth always has to run, and we use it to bootstrap our time estimation.
		depthStart := time.Now()
		bestAction, bestBoard, bestScore = ab.searchToMaxDepth(nil, board, 1, board.NextPlayer)
		lastDepthDuration := time.Since(depthStart)

		// Loop iteratively increasing the depth:
		currentDepth := 2
		for {
			if ab.maxDepth > 0 && currentDepth > ab.maxDepth {
				break
			}
			depthStart = time.Now()
			if deadline.Sub(depthStart) < 5*lastDepthDuration {
				// Not enough time for next iteration, just interrupt.
				break
			}
			ctx, ctxCancel := context.WithDeadline(context.Background(), deadline)
			newAction, newBoard, newScore := ab.searchToMaxDepth(ctx, board, currentDepth, board.NextPlayer)
			interrupted := ctx.Err() != nil
			ctxCancel()
			if interrupted {
				break
			}
			bestAction, bestBoard, bestScore = newAction, newBoard, newScore
			lastDepthDuration = time.Since(depthStart)
			currentDepth++
		}
		klog.V(1).Infof("Move #%d searched depth %d within %s (maxTime=%s)", board.MoveNumber, currentDepth-1, time.Since(start), ab.maxTime)
	}

	// TODO: implement maxTime by interactively increasing the depth in the search, until the time expires.
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
		batchScores := ab.scorer.BatchScore([]*Board{board})
		fmt.Printf("Best action found: %s - shallow score=%.2f, αβ-score=%.2f\n\n",
			bestAction, batchScores[0], bestScore)
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
func (ab *Searcher) searchToMaxDepth(ctx context.Context, board *Board, maxDepth int, mainPlayer PlayerNum) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	alpha := float32(-math.MaxFloat32)
	beta := float32(-math.MaxFloat32)
	addNoise := ab.randomness > 0 && (ab.maxMoveRandomness <= 0 || board.MoveNumber <= ab.maxMoveRandomness)
	bestAction, bestBoard, bestScore = ab.recursion(ctx, board, maxDepth, mainPlayer, alpha, beta, addNoise)
	return
}

var muLogBoard sync.Mutex

// recursion of the alpha-beta pruning algorithm, with depthLeft plies to go.
// mainPlayer is the player that initiated the search -- some values are not symmetric, so we need this.
func (ab *Searcher) recursion(ctx context.Context, board *Board, depthLeft int, mainPlayer PlayerNum, alpha, beta float32, addNoise bool) (
	bestAction Action, bestBoard *Board, bestScore float32) {
	isLeaf := depthLeft <= 1
	if ctx != nil && ctx.Err() != nil {
		// ctx interrupted/expired.
		return
	}

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
	// Also, there are no discounts for guaranteed wins.
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

		// Switch the score of draws.
		if ab.drawScore != 0 {
			for actionIdx, board := range newBoards {
				if board.Draw() {
					newScore := ab.drawScore
					if board.NextPlayer != mainPlayer {
						newScore = -newScore
					}
					scores[actionIdx] = newScore
				}
			}
		}

		// Pick best:
		bestScore, bestActionIdx = pickBest(scores)
		bestBoard = newBoards[bestActionIdx]
		bestAction = actions[bestActionIdx]
		return
	}

	// Find order from the best scoring first.
	bestScore = float32(-math.MaxFloat32)
	bestBoard = nil
	bestAction = Action{}
	ordering := generics.SliceOrdering(scores, true) // Reverse order by score.
	for _, actionIdx := range ordering {
		if ctx != nil && ctx.Err() != nil {
			// Deadline expired.
			return
		}
		if searchers.IdleChan != nil {
			// Wait for an "idle" signal before each search: only used in WASM.
			<-searchers.IdleChan
		}

		// Only follows recursion if this action doesn't end the match.
		if !newBoards[actionIdx].IsFinished() {
			// Runs alphaBeta for opponent player, so the alpha/beta are reversed.
			_, _, score := ab.recursion(ctx, newBoards[actionIdx], depthLeft-1, mainPlayer, beta, alpha, addNoise)

			// Apply discount to non-winning scores.
			if math32.Abs(score) < ai.WinGameScore {
				score *= ab.discount
			}

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
func executeAndScoreActions(board *Board, scorer ai.BatchValueScorer) (newBoards []*Board, scores []float32) {
	actions := board.Derived.Actions
	scores = make([]float32, len(actions))
	newBoards = make([]*Board, len(actions))

	// Pre-score actions that lead to end-game.
	boardsToScore := make([]*Board, 0, len(actions))
	newBoards = board.TakeAllActions()
	hasWinning := 0
	for actionIdx, newBoard := range newBoards {
		if isEnd, score := ai.IsEndGameAndScore(newBoard); isEnd {
			// End game is treated differently.
			score = -score // Score for board.NextPlayer, not newBoards[actionIdx].NextPlayer
			if score == ai.WinGameScore {
				hasWinning++
			}
			scores[actionIdx] = score
		} else {
			boardsToScore = append(boardsToScore, newBoard)
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
		scored := scorer.BatchScore(boardsToScore)
		scoredIdx := 0
		for actionIdx, newBoard := range newBoards {
			if !newBoard.IsFinished() {
				// Score for board.NextPlayer, not newBoards[actionIdx].NextPlayer, hence
				// we take the inverse here.
				scores[actionIdx] = -1 * 0.999 * scored[scoredIdx] // Evaluation score capped to +/- 0.999, to avoid mixing with actual game won/lost.
				scoredIdx++
			}
		}
	}
	return
}

// pickBest returns the best score and its index.
// If there are ties, it picks randomly among the best indices.
func pickBest(scores []float32) (bestScore float32, bestIdx int) {
	numBestScores := 0
	bestScore = float32(-math.MaxFloat32)
	bestIdx = -1
	for scoreIdx, score := range scores {
		if score < bestScore {
			continue
		}
		if score > bestScore {
			bestScore = score
			bestIdx = scoreIdx
			numBestScores = 1
			continue
		}
		// It's a tie, so we randomly keep the current one or pick the new one.
		numBestScores++
		if rand.Intn(numBestScores) == 0 {
			bestIdx = scoreIdx
		}
	}
	return
}
