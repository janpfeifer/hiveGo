package mcts

// Monte Carlo Tree Search implementation for Alpha-Zero algorith. First very good description
// in a post from Surag Nair, in https://web.stanford.edu/~surag/posts/alphazero.html

import (
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/internal/ai/linear"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/janpfeifer/hiveGo/ascii_ui"
)

var (
	_ = log.Printf
	_ = fmt.Printf
)

const (
	DECAY                     = float32(0.999)
	DEPTH_CHECK_MAX_ABS_SCORE = 5
	EPSILON                   = float64(1e-8)
)

var (
	flag_mctsUseLinearScore = flag.Bool("mcts_use_linear_score",
		false, "If set, it will use a linear model for scoring (but "+
			"not for the action probabilities)")
)

type mctsSearcher struct {
	maxDepth                   int
	maxTime                    time.Duration
	maxTraverses, minTraverses int
	maxAbsScore                float32 // Max absolute score, value above that interrupt the search.
	cPuct                      float32 // Degree of exploration of alpha-zero.

	// Useful so that it doesn't play exactly the same match every time. Applied
	// only on the first level of the MCTS traversal.
	randomness float32

	// If to explore paths in parallel. Not yet supported.
	parallelized bool

	// BoardScorer to use during search.
	scorer ai.BatchBoardScorer

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
	// Parent information is only used for debugging.
	parent          *cacheNode
	parentActionIdx int

	// Board, actions, children boards and children base scores.
	board   *Board
	actions []Action

	// Predictions: Score for board.NextPlayer and probability of each action.
	score        float32
	actionsProbs []float32

	// Children cacheNodes.
	cacheNodes []*cacheNode

	// How many times each of the paths have been traversed.
	count      []int
	totalCount int // Sum of all count.

	// Sum of the scores returned by the MCTS for each path taken.
	sumMCScores []float32 // Sum of the scores on each of the traversals at the given path.

	// Estimation of Q(s, a), where s is the state (board here), and a is the action.
	// This is also renormalized to be between -1 (loosing) and 1 (winning)
	Q []float32

	// Lock for updates.
	mu sync.Mutex
}

// modelScoreToQ converts "score" scale (from -10 to 10) to "Q" scale (-1 to 1).
func modelScoreToQ(score float32) float32 {
	return score / 10.0
}

// root cache node indicates if this is the root of the MCTS. This is used
// to decide if randomness should be used.
func newCacheNode(mcts *mctsSearcher, stats *matchStats, b *Board, root bool) *cacheNode {
	numActions := len(b.Derived.Actions)
	numChildren := numActions
	if numActions == 0 {
		numChildren = 1
	}

	cn := &cacheNode{
		board:       b,
		actions:     b.Derived.Actions,
		cacheNodes:  make([]*cacheNode, numChildren),
		count:       make([]int, numActions),
		sumMCScores: make([]float32, numActions),
		Q:           make([]float32, numActions),
	}
	if stats != nil {
		stats.numCacheNodes++
	}
	cn.score, cn.actionsProbs = mcts.scorer.Score(b, true)

	// Sanity check:
	var sumProbs float32
	for ii, prob := range cn.actionsProbs {
		if prob < -1e-3 {
			ui := ascii_ui.NewUI(true, false)
			fmt.Println()
			ui.PrintBoard(cn.board)
			fmt.Printf("Available actions: %v", cn.board.Derived.Actions)
			fmt.Printf("Probabilities: %v", cn.actionsProbs)
			log.Panicf("Negative probability for action %v, prob=%g",
				cn.actions[ii], prob)
		}
		sumProbs += prob
	}
	if len(cn.actions) > 1 && abs32(sumProbs-1.0) > 1e-3 {
		ui := ascii_ui.NewUI(true, false)
		fmt.Println()
		ui.PrintBoard(cn.board)
		fmt.Printf("Available actions: %v", cn.board.Derived.Actions)
		fmt.Printf("Probabilities: %v", cn.actionsProbs)
		log.Panicf("Sum of probabilities=%g != 1.0", sumProbs)
	}

	if *flag_mctsUseLinearScore {
		newScore, _ := linear.TrainedBest.Score(b, false)
		cn.score = newScore
	}
	if root && mcts.randomness > 0 {
		for ii := range cn.actionsProbs {
			cn.actionsProbs[ii] += float32(rand.NormFloat64()) * mcts.randomness
		}
	}
	for ii := range cn.Q {
		cn.Q[ii] = modelScoreToQ(cn.score)
	}
	return cn
}

// Step steps into the index's cacheNode under the current one. If it doesn't exist,
// it creates a new one.
// Setting `index == -1` is valid if there are no valid actions. In this case a single
// child node is considered and used.
func (cn *cacheNode) Step(mcts *mctsSearcher, stats *matchStats, index int, log bool) *cacheNode {
	cn.mu.Lock()
	defer cn.mu.Unlock()

	cnIdx := index
	if index < 0 {
		cnIdx = 0
	}
	newCN := cn.cacheNodes[cnIdx]
	if newCN == nil {
		var action Action
		if index >= 0 {
			action = cn.actions[index]
		} else {
			action = SkipAction
		}
		newCN = newCacheNode(mcts, stats, cn.board.Act(action), false)
		cn.cacheNodes[cnIdx] = newCN
		if log {
			klog.V(1).Infof("Create new cacheNode with action %s", action)
		}
	}
	return newCN
}

// Clear MCTS related scores -- but not the action probabilities and board value
// estimated by the NN.
func (cn *cacheNode) RecursivelyClearMCTSScores(mcts *mctsSearcher) {
	for ii := range cn.actions {
		cn.count[ii] = 0
		cn.sumMCScores[ii] = 0
		cn.Q[ii] = 0
	}
	cn.totalCount = 0

	for _, childCN := range cn.cacheNodes {
		if childCN != nil {
			childCN.RecursivelyClearMCTSScores(mcts)
		}
	}
}

// Sample picks the next step, with probability weighted
// by the estimated upper bound.
func (cn *cacheNode) Sample(mcts *mctsSearcher) int {
	cn.mu.Lock()
	defer cn.mu.Unlock()

	if len(cn.actions) == 0 {
		return -1
	}
	if len(cn.actions) == 1 {
		return 0
	}
	best := -1
	bestActionAdjustedQ := float32(-1e6)
	globalFactor := mcts.cPuct * float32(math.Sqrt(float64(cn.totalCount)+EPSILON))
	hasLosses := false

	for ii := range cn.actions {
		actionCN := cn.cacheNodes[ii]
		actionAdjustedQ := cn.Q[ii] + globalFactor*cn.actionsProbs[ii]/(1+float32(cn.count[ii]))
		if actionCN != nil {
			if actionCN.board.IsFinished() {
				if actionCN.score > 0 {
					// Opponent wins (since actionCN is the board with the opponent turn),
					// so no need to sample this one. Odd, since the player would
					// cause their own loss.
					hasLosses = true
					continue
				} else if actionCN.score > 0 {
					// Player wins, so it would simply greedly take this action.
					best = ii
					bestActionAdjustedQ = actionAdjustedQ
					break
				}
			}
		}
		klog.V(4).Infof("Sampling: adjustedQ=%g, Q=%g, probs=%.2g", actionAdjustedQ,
			cn.Q[ii], cn.actionsProbs[ii])
		if actionAdjustedQ > bestActionAdjustedQ {
			best = ii
			bestActionAdjustedQ = actionAdjustedQ
		}
	}
	if best == -1 {
		if hasLosses {
			klog.V(1).Infof("Sampling: all actions lead to loss, taking the first.")
			return 0
		}
		log.Panicf("Not able to sample a move, likely probabilities are NaN.")
	}
	if cn.parent == nil {
		klog.V(3).Infof("Sampled %s: adjustedQ=%g, Q=%g, probs=%.2g%%",
			cn.actions[best], bestActionAdjustedQ, cn.Q[best], 100.0*cn.actionsProbs[best])
	}
	return best
}

func (cn *cacheNode) FindBestScore(mcts *mctsSearcher) (
	bestIdx int, bestScore float32, actionsLabels []float32) {
	if len(cn.actions) <= 1 {
		// There is either one or none actions. Index will be set to 0 (first action)
		// or -1 (represents a SkipAction) respectively, and the score should be the
		// negative of the score for the opponent at the next action.
		bestIdx = len(cn.actions) - 1
		bestScore = cn.score
		if cn.cacheNodes[0] != nil {
			_, childScore, _ := cn.cacheNodes[0].FindBestScore(mcts)
			bestScore = -childScore
		}
		if len(cn.actions) == 1 {
			actionsLabels = []float32{1}
		}
		return
	}

	// If nothing was visited, simply return the current estimated score
	// and the action with largest probability.
	actionsLabels = make([]float32, len(cn.actions))
	if cn.totalCount == 0 {
		klog.Errorf("MCTS.FindBestNode() called with no visits having been made.")
		bestScore = cn.score
		bestIdx = 0
		bestProb := cn.actionsProbs[0]
		actionsLabels[0] = 1.0 / float32(len(cn.actions))
		for ii := 1; ii < len(cn.actions); ii++ {
			actionsLabels[ii] = actionsLabels[0]
			if cn.actionsProbs[ii] > bestProb {
				bestIdx = ii
				bestProb = cn.actionsProbs[ii]
			}
		}
		return
	}

	// Select best action based on count of visits (see paper), and in case of ties, break by
	// Q(s,a) estimation. Final score is given by mean of the sumMCScores.
	bestIdx = -1
	bestCount := -1
	totalCount := float32(cn.totalCount)
	for ii := 0; ii < len(cn.actions); ii++ {
		if cn.count[ii] > bestCount || (cn.count[ii] == bestCount && cn.Q[ii] > bestScore) {
			bestIdx = ii
			bestCount = cn.count[ii]
			bestScore = cn.Q[ii]
		}
		actionsLabels[ii] = float32(cn.count[ii]) / totalCount
	}
	bestScore = cn.sumMCScores[bestIdx] / float32(cn.count[bestIdx])
	return
}

func abs32(x float32) float32 { return float32(math.Abs(float64(x))) }

func (cn *cacheNode) recursiveLogTraverse(actionIdx int, parts []string) ([]string, float32) {
	if cn.parent != nil {
		if actionIdx < 0 {
			parts = append(parts, SkipAction.String())
		} else {
			parts = append(parts, cn.actions[actionIdx].String())
		}
		var mult float32
		parts, mult = cn.parent.recursiveLogTraverse(cn.parentActionIdx, parts)
		return parts, -mult * DECAY
	}
	return parts, -1
}

func (cn *cacheNode) logTraverse(score float32, stopReason string) {
	var parts []string
	if cn.parent != nil {
		var mult float32
		parts, mult = cn.parent.recursiveLogTraverse(cn.parentActionIdx, parts)
		score *= -mult
	}
	for ii := len(parts)/2 - 1; ii >= 0; ii-- {
		jj := len(parts) - ii - 1
		parts[ii], parts[jj] = parts[jj], parts[ii]
	}
	klog.Infof("Traverse (%s): score=%.6f, path=[%s]", stopReason, score, strings.Join(parts, "], ["))
}

// Traverse traverses the game tree up to the given depth, and returns the
// expected score returned by the leaf node (or deepest node) visited.
// The score is with respect to the player playing (board.NextPlayer) at the node cn.
func (cn *cacheNode) Traverse(mcts *mctsSearcher, stats *matchStats, depthLeft int) float32 {
	// Checks end of game conditions.
	if cn.board.IsFinished() {
		_, score := ai.EndGameScore(cn.board)
		if klog.V(3) {
			cn.logTraverse(score, "end game")
		}
		return score
	}

	// Checks max depth reached.
	if depthLeft == 0 {
		if klog.V(3) {
			cn.logTraverse(cn.score, "max depth")
		}
		return cn.score
	}

	// Checks if score threshold is reached.
	if depthLeft < mcts.maxDepth-DEPTH_CHECK_MAX_ABS_SCORE {
		if abs32(cn.score) >= mcts.maxAbsScore {
			if klog.V(3) {
				cn.logTraverse(cn.score, "max abs score")
			}
			return cn.score
		}
	}

	// Sample new action.
	actionIdx := cn.Sample(mcts)
	nextCN := cn.Step(mcts, stats, actionIdx, false)
	nextCN.parent = cn
	nextCN.parentActionIdx = actionIdx
	score := -nextCN.Traverse(mcts, stats, depthLeft-1) * DECAY

	// If there are no actions, or only one action, we don't keep tabs, since
	// there are no options anyway.
	if len(cn.actions) <= 1 {
		return score
	}

	// Propagate back the score.
	cn.mu.Lock()
	cn.totalCount++
	cn.count[actionIdx]++
	cn.sumMCScores[actionIdx] += score
	meanScore := cn.sumMCScores[actionIdx] / float32(cn.count[actionIdx])
	cn.Q[actionIdx] = modelScoreToQ(meanScore)
	cn.mu.Unlock()
	if depthLeft == mcts.maxDepth {
		klog.V(3).Infof("Traverse[%s]: score=%.2f, Q=%.2f", cn.actions[actionIdx], score, cn.Q[actionIdx])
	}
	return score
}

// runMCTS runs MCTS for the given specifications on the cacheNode.
func (mcts *mctsSearcher) runOnCN(stats *matchStats, cn *cacheNode) {
	// Sample while there is time.
	if len(cn.actions) > 1 {
		start := time.Now()
		count := 0

		// Handle parallelism.
		// TODO: parallelized forced to false for now (see NewMCSTSearcher). Still need to implement
		//   locks such that no traverse share paths. Also the algorithm should change since it changes the
		//   counts

		// Loop over traverses.
		for (time.Since(start) < mcts.maxTime || count < mcts.minTraverses) &&
			count < mcts.maxTraverses {
			cn.Traverse(mcts, stats, mcts.maxDepth)
			count++
		}
		klog.V(1).Infof("MCTS traverses done: %d", count)
	}
}

func (mcts *mctsSearcher) measuredRunOnCN(stats *matchStats, cn *cacheNode) {
	beforeCacheNodes := stats.numCacheNodes
	start := time.Now()
	mcts.runOnCN(stats, cn)
	elapsedTime := time.Since(start)
	searchCacheNodes := stats.numCacheNodes - beforeCacheNodes

	klog.V(1).Infof("States searched in this move:    \t%d CacheNodes",
		searchCacheNodes)
	cacheNodesPerSec := float64(searchCacheNodes) / float64(elapsedTime.Seconds())
	klog.V(1).Infof("Rate of evaluations in this move:\t%.1f CacheNodes/s",
		cacheNodesPerSec)

	klog.V(1).Infof("States searched in match so far: \t%d CacheNodes",
		stats.numCacheNodes)
}

// Search implements the Searcher interface.
func (mcts *mctsSearcher) Search(b *Board) (
	action Action, board *Board, score float32, actionsLabels []float32) {
	return mcts.searchWithStats(nil, b)
}

type sortableProbsActions struct {
	indices []int
	probs   []float32
}

func (pa *sortableProbsActions) Swap(i, j int) {
	pa.indices[i], pa.indices[j] = pa.indices[j], pa.indices[i]
}
func (pa *sortableProbsActions) Len() int {
	return len(pa.indices)
}
func (pa *sortableProbsActions) Less(i, j int) bool {
	return pa.probs[pa.indices[i]] > pa.probs[pa.indices[j]]
}

func logTopActionProbs(labelProbs []float32, actions []Action, prevProbs, scores []float32) {
	if len(scores) <= 1 || len(actions) <= 1 {
		return
	}
	var sumProbs float32
	for _, prob := range prevProbs {
		sumProbs += prob
	}
	klog.Infof("  Model previous probabilities: [%v], sum(+rand)=%.3g", prevProbs, sumProbs)
	max, min := prevProbs[0], prevProbs[0]
	for _, prob := range prevProbs {
		if prob > max {
			max = prob
		} else if prob < min {
			min = prob
		}
	}
	klog.Infof("Difference max(prob)-min(prob)=%.4g%%",
		100.0*(max-min))
	sorted := sortableProbsActions{
		indices: make([]int, len(labelProbs)),
		probs:   labelProbs,
	}
	for ii := range sorted.indices {
		sorted.indices[ii] = ii
	}
	sort.Sort(&sorted)
	for ii := 0; ii < len(sorted.indices) && ii < 5; ii++ {
		idx := sorted.indices[ii]
		if labelProbs[idx] < 0.02 {
			break
		}
		klog.Infof("  Action %s: new probability %.2f%%, prev probability %.2f%%, Q-score=%.2g",
			actions[idx], 100*labelProbs[idx], 100*prevProbs[idx], scores[idx])
	}
	klog.Infof("  New probabilities: [%v]", labelProbs)
}

func (mcts *mctsSearcher) searchWithStats(stats *matchStats, b *Board) (
	action Action, board *Board, score float32, actionsLabels []float32) {
	cn := newCacheNode(mcts, stats, b, true)
	if klog.V(1) {
		// Measure time and boards evaluated.
		if stats == nil {
			stats = &matchStats{}
		}
		mcts.measuredRunOnCN(stats, cn)
	} else {
		mcts.runOnCN(stats, cn)
	}

	var actionIdx int
	actionIdx, score, actionsLabels = cn.FindBestScore(mcts)
	board = nil
	if actionIdx >= 0 {
		action = cn.actions[actionIdx]
		if cn.cacheNodes[actionIdx] != nil {
			board = cn.cacheNodes[actionIdx].board
		}
		if klog.V(2) {
			fmt.Println()
			ui := ascii_ui.NewUI(true, false)
			ui.PrintBoard(cn.board)
			fmt.Println()
			klog.Infof("Search Move #%d (p%d), %d actions available, baseline is %.2g%%:",
				cn.board.MoveNumber, cn.board.NextPlayer,
				len(cn.actions), 100.0/float64(len(cn.actions)))
			logTopActionProbs(actionsLabels, cn.actions, cn.actionsProbs, cn.Q)
		}

	} else {
		action = SkipAction
		board = cn.cacheNodes[0].board
	}
	if board == nil {
		board = cn.board.Act(action)
	}
	klog.V(1).Infof("Selected action %s, score %.2f", action, score)
	return
}

// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions)+1, and
// len(actionsLabels) == len(actions).
func (mcts *mctsSearcher) ScoreMatch(b *Board, actions []Action) (
	scores []float32, actionsLabels [][]float32) {
	stats := &matchStats{}
	ui := ascii_ui.NewUI(true, false)

	// For re-scoring matches, there is no "root", since cacheNode is not
	// recreate at every action. Also because when rescoring randomness
	// is not desired.
	cn := newCacheNode(mcts, stats, b, false)

	for matchActionsIdx, action := range actions {
		// Search current node.
		if klog.V(1) {
			// Measure time and boards evaluated.
			mcts.measuredRunOnCN(stats, cn)
		} else {
			mcts.runOnCN(stats, cn)
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
		if klog.V(2) {
			fmt.Println()
			klog.Infof("ScoreMatch Move #%d (%d to go), player %d has the turn:",
				cn.board.MoveNumber, len(actions)-matchActionsIdx, cn.board.NextPlayer)
			ui.PrintBoard(cn.board)
			logTopActionProbs(boardActionsLabels, cn.actions, cn.actionsProbs, cn.Q)
			fmt.Println()
		}

		// The action taken may be different than the best action, specially
		// as the model evolves.
		var playedIdx int
		if action.IsSkipAction() {
			playedIdx = -1
		} else {
			playedIdx = cn.board.FindActionDeep(action)
			klog.V(2).Infof("Actually played: %s, prob=%.2g%%, prev_prob=%.2g%%, Q-score=%f",
				cn.actions[playedIdx], boardActionsLabels[playedIdx]*100, cn.actionsProbs[playedIdx], cn.Q[playedIdx])
		}
		cn = cn.Step(mcts, stats, playedIdx, true)
		if isEnd, score := ai.EndGameScore(cn.board); isEnd {
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
