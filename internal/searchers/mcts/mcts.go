// Package mcts is a Monte Carlo Tree Search implementation of searchers.Searcher for
// the Alpha-Zero algorithm.
//
// References used, since the original paper doesn't actually provide the formulas:
//
//   - https://suragnair.github.io/posts/alphazero.html by Surag Nair
//   - https://web.stanford.edu/class/archive/cs/cs221/cs221.1196/sections/Section5.pdf
//
// AlphaZero original paper -- that mostly talks about its successes but not the actual
// formula:
//
//   - Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
//     https://arxiv.org/abs/1712.01815
//
// AlphaGo Zero has more details, but too many for someone who simply wants to understand how
// it is done:
//
//   - https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ
package mcts

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"log"
	"math"
	"time"

	"github.com/janpfeifer/hiveGo/internal/ui/cli"
)

var (
	_ = log.Printf
	_ = fmt.Printf
)

const (
	DECAY                     = float32(0.999)
	DEPTH_CHECK_MAX_ABS_SCORE = 5
)

type mctsSearcher struct {
	maxDepth                   int
	maxTime                    time.Duration
	maxTraverses, minTraverses int
	maxAbsScore                float32 // Max absolute score, value above that interrupt the search.
	cPuct                      float32 // Degree of exploration of alpha-zero.

	// temperature (usually represented as the greek letter τ) is an exponent applied
	// to the counts used in the policy distribution (π) formula. If set to zero, it will
	// always take the best estimate action. AlphaZero Go uses 1 for the first 30 moves.
	// Larger models will make the play more random.
	temperature float32

	// If to explore paths in parallel. Not yet supported.
	parallelized bool

	// ValueScorer to use during search.
	scorer ai.PolicyScorer

	// Player parameter that indicates that MCTS was selected.
	useMCTS bool
}

type matchStats struct {
	// Number of candidate nodes generated during search: used for performance measures.
	numCacheNodes int
}

func (mcts *mctsSearcher) Clone() *mctsSearcher {
	r := &mctsSearcher{}
	*r = *mcts
	return r
}

// cacheNode holds information about the possible actions of a board.
type cacheNode struct {
	// Board, actions, children boards and children base scores.
	board *Board

	// actionsProbs are the model actions probabilities.
	actionsProbs []float32

	// Children cacheNodes.
	cacheNodes []*cacheNode

	// N is the count per action of which paths have been traversed.
	// If nil, it is assumed to be 0 for all actions.
	N []int

	// sumN holds the sum of all values of N.
	sumN int

	// sumScores of the score of taking the corresponding action at the current board.
	// If N[a] > 0, we have $Q(s, a) = sumScores[a]/N[a]$.
	sumScores []float32
}

// newCacheNode for the given board position and updated matchStats.
// root indicates this is a root node for the search, this decides whether the temperature should be used.
func (mcts *mctsSearcher) newCacheNode(b *Board, stats *matchStats) (*cacheNode, error) {
	if b.IsFinished() {
		return nil, errors.Errorf("can't create cacheNode for a finished board state")
	}
	numActions := len(b.Derived.Actions)
	cn := &cacheNode{
		board:      b,
		cacheNodes: make([]*cacheNode, numActions),
		N:          make([]int, numActions),
		sumScores:  make([]float32, numActions),
	}
	if stats != nil {
		stats.numCacheNodes++
	}
	cn.actionsProbs = mcts.scorer.PolicyScore(b)

	// Sanity check:
	var sumProbs float32
	for _, prob := range cn.actionsProbs {
		if prob < 0 {
			klog.Errorf("Board has negative action probability %g !?", prob)
			ui := cli.New(true, false)
			fmt.Println()
			ui.PrintBoard(cn.board)
			fmt.Printf("Available actions: %v", cn.board.Derived.Actions)
			fmt.Printf("Probabilities: %v", cn.actionsProbs)
			return nil, errors.Errorf("board scorer returned negative probability %g for board position", prob)
		}
		sumProbs += prob
	}
	if math.Abs(float64(sumProbs-1.0)) > 1e-3 {
		ui := cli.New(true, false)
		fmt.Println()
		ui.PrintBoard(cn.board)
		fmt.Printf("Available actions: %v", cn.board.Derived.Actions)
		fmt.Printf("Probabilities: %v", cn.actionsProbs)
		return nil, errors.Errorf("sum of probabilities=%g != 1.0", sumProbs)
	}

	return cn, nil
}

// SearchSubtree rooted on cn, expanding one board.
//
// It returns the new sampled score for the "next player" (to play) of cacheNode's board.
//
// Notice it doesn't return the score estimate (Q) of all samples in the sub-tree, but simply
// the score of the individual new sample (the value returned by the scorer on the leaf-node
// of the recursion).
//
// This is the core of the AlphaZero/MCTS algorithm, based on the estimated
// upper bounds of each possible action.
func (mcts *mctsSearcher) SearchSubtree(cn *cacheNode, stats *matchStats) (score float32, err error) {
	// Find the action with the best upper confidence (U in the description).
	bestAction := -1
	bestUpperConfidence := float32(math.Inf(-1))
	globalFactor := mcts.cPuct * float32(math.Sqrt(float64(cn.sumN)))
	for actionIdx, numVisits := range cn.N {
		var Q float32 // 0 if we haven't subsampled it yet.
		if numVisits > 0 {
			Q = cn.sumScores[actionIdx] / float32(numVisits)
		}
		upperConfidence := Q + globalFactor*cn.actionsProbs[actionIdx]/float32(1+numVisits)
		if upperConfidence > bestUpperConfidence {
			bestAction = actionIdx
			bestUpperConfidence = upperConfidence
		}
	}

	// For the first time an action is considered, just get the plain score estimate
	// for the new board.
	if cn.N[bestAction] == 0 {
		// Notice TakeAllActions is cached in the board.
		newBoard := cn.board.TakeAllActions()[bestAction]
		if isEnd, endScore := ai.IsEndGameAndScore(newBoard); isEnd {
			// TODO: add some optimization/check if we are repeatedly sampling a game that is won or lost,
			// 	and there is no alternative play, to accelerate end-game.
			score = -endScore
		} else {
			score = -mcts.scorer.Score(newBoard)
		}
		cn.N[bestAction] = 1
		cn.sumN++
		cn.sumScores[bestAction] += score
		return
	}

	// If not the first time we sample the action, make sure we have a corresponding
	// cacheNode for it, expanding the tree.
	if cn.cacheNodes[bestAction] == nil {
		newBoard := cn.board.TakeAllActions()[bestAction]
		if isEnd, endScore := ai.IsEndGameAndScore(newBoard); isEnd {
			// Return immediately and don't create a cacheNode.
			score = -endScore
			cn.N[bestAction] = 1
			cn.sumN++
			cn.sumScores[bestAction] += score
			return
		}

		cn.cacheNodes[bestAction], err = mcts.newCacheNode(newBoard, stats)
		if err != nil {
			return
		}
	}

	// Recursively sample value of the best action.
	score, err = mcts.SearchSubtree(cn.cacheNodes[bestAction], stats)
	score = -score
	if err != nil {
		return
	}
	cn.sumScores[bestAction] += score
	cn.N[bestAction]++
	cn.sumN++
	return
}

// Search implements searchers.Searcher API.
//
// It returns the expected best action, board, and score estimate of the given best action.
//
// TODO: implement parallelism in MCTS.
func (mcts *mctsSearcher) Search(board *Board) (bestAction Action, bestBoard *Board, bestScore float32, err error) {
	var rootCacheNode *cacheNode
	var stats matchStats
	rootCacheNode, err = mcts.newCacheNode(board, &stats)
	if err != nil {
		return
	}

	// Keep sampling until the time is over.
	startTime := time.Now()
	var elapsed time.Duration
	for {
		_, err = mcts.SearchSubtree(rootCacheNode, &stats)
		if err != nil {
			return
		}
		elapsed = time.Since(startTime)
		if elapsed > mcts.maxTime {
			break
		}
	}

	// Log performance.
	if klog.V(1).Enabled() {
		cacheNodeRate := float64(stats.numCacheNodes) / elapsed.Seconds()
		klog.Infof("Search at move #%d: %.2f nodes/s", board.MoveNumber, cacheNodeRate)
	}

	// Select best action and its estimate.
	bestActionIdx, mostVisits := -1, -1
	for actionIdx, nVisits := range rootCacheNode.N {
		if nVisits > mostVisits {
			mostVisits = nVisits
			bestActionIdx = actionIdx
		}
	}
	bestAction = board.Derived.Actions[bestActionIdx]
	bestBoard = board.TakeAllActions()[bestActionIdx]
	// bestScore uses the Q estimate.
	bestScore = rootCacheNode.sumScores[bestActionIdx] / float32(mostVisits)
	return
}

/*
// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions)+1, and
// len(actionsLabels) == len(actions).
func (mcts *mctsSearcher) ScoreMatch(b *Board, actions []Action) (
	scores []float32, actionsLabels [][]float32, err error) {
	stats := &matchStats{}
	ui := cli.New(true, false)

	// For re-scoring matches, there is no "root", since cacheNode is not
	// recreate at every action. Also because when rescoring temperature
	// is not desired.
	cn, err := mcts.newCacheNode(b, false, stats)
	if err != nil {
		return nil, nil, err
	}

	for matchActionsIdx, action := range actions {
		// Search current node.
		if klog.V(1).Enabled() {
			// Measure time and boards evaluated.
			err = mcts.measuredRunOnCN(stats, cn)
		} else {
			err = mcts.runOnCN(stats, cn)
		}
		if err != nil {
			return nil, nil, err
		}

		// Score of this node, is the score of the best action.
		_, score, boardActionsLabels := cn.FindBestScore(mcts)
		scores = append(scores, score)
		actionsLabels = append(actionsLabels, boardActionsLabels)
		if len(boardActionsLabels) != cn.board.NumActions() {
			fmt.Println()
			ui.PrintBoard(cn.board)
			fmt.Printf("Available actions: %v", cn.board.Derived.Actions)
			log.Panicf("Number of labels (%d) different than number of actions (%d) for board on move %d",
				len(boardActionsLabels), cn.board.NumActions(), cn.board.MoveNumber)
		}
		if klog.V(2).Enabled() {
			fmt.Println()
			klog.Infof("ScoreMatch Move #%d (%d to go), player %d has the turn:",
				cn.board.MoveNumber, len(actions)-matchActionsIdx, cn.board.NextPlayer)
			ui.PrintBoard(cn.board)
			logTopActionProbs(boardActionsLabels, cn.actions, cn.actionsProbs, cn.sumScores)
			fmt.Println()
		}

		// The action taken may be different from the best action, specially
		// as the model evolves.
		var playedIdx int
		if action.IsSkipAction() {
			playedIdx = -1
		} else {
			playedIdx = cn.board.FindActionDeep(action)
			klog.V(2).Infof("Actually played: %s, prob=%.2g%%, prev_prob=%.2g%%, Q-score=%f",
				cn.actions[playedIdx], boardActionsLabels[playedIdx]*100, cn.actionsProbs[playedIdx], cn.sumScores[playedIdx])
		}
		cn, err = cn.Step(mcts, stats, playedIdx, true)
		if err != nil {
			return nil, nil, err
		}
		if isEnd, score := ai.IsEndGameAndScore(cn.board); isEnd {
			scores = append(scores, score)
			return
		}
		// Clear scores found from previous level search, since not it may go deeper.
		// The boards, scores and probabilities.
		cn.RecursivelyClearMCTSScores(mcts)
	}

	// Add the final board score, if the match hasn't ended yet.
	_, score, _ := cn.FindBestScore(mcts)
	scores = append(scores, score)
	return
}
*/
