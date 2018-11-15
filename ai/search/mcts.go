package search

// Monte Carlo Tree Search implementation, with a hacked version of UCT
// (Upper Confidence Bound for Trees).

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	. "github.com/janpfeifer/hiveGo/state"
)

var (
	_ = log.Printf
	_ = fmt.Printf
)

const epsilon = float64(1e-8)

const DEPTH_CHECK_MAX_ABS_SCORE = 5

type mctsSearcher struct {
	maxDepth     int
	maxTime      time.Duration
	maxTraverses int
	maxAbsScore  float32 // Max absolute score, value above that interrupt the search.

	// Number of candidate nodes generated during search: used for performance measures.
	numCacheNodes int

	scorer       ai.BatchScorer
	randomness   float32
	parallelized bool

	// Cache previous searches on the current tree. Reused by score match.
	reuseCN *cacheNode
}

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPrunning.
func NewMonteCarloTreeSearcher(maxDepth int, maxTime time.Duration,
	maxTraverses int, maxAbsScore float32,
	scorer ai.BatchScorer, randomness float64, parallelized bool) Searcher {
	if parallelized {
		glog.Error("MCTS does not yet support parallelized run.")
		parallelized = false
	}
	return &mctsSearcher{
		maxDepth:     maxDepth,
		maxTime:      maxTime,
		maxTraverses: maxTraverses,
		maxAbsScore:  maxAbsScore,

		numCacheNodes: 0,

		scorer:       scorer,
		randomness:   float32(randomness),
		parallelized: parallelized,
	}
}

// cacheNode holds information about the possible actions of a board.
type cacheNode struct {
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

func newCacheNode(mcts *mctsSearcher, b *Board) *cacheNode {
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
	mcts.numCacheNodes++
	cn.score, cn.actionsProbs = mcts.scorer.Score(b)
	return cn
}

// Step steps into the index's cacheNode under the current one. If it doesn't exist,
// it creates a new one.
// Setting `index == -1` is valid if there are no valid actions. In this case a single
// child node is considered and used.
func (cn *cacheNode) Step(mcts *mctsSearcher, index int) *cacheNode {
	cn.mu.Lock()
	defer cn.mu.Unlock()

	cnIdx := index
	if index < 0 {
		cnIdx = 0
	}
	newCN := cn.cacheNodes[cnIdx]
	if newCN == nil {
		var action Action
		if index > 0 {
			action = cn.actions[index]
		} else {
			action = SKIP_ACTION
		}
		newCN = newCacheNode(mcts, cn.board.Act(action))
		cn.cacheNodes[cnIdx] = newCN
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
// by expected score. Not used if useUCT is set.
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
	bestActionScore := float32(-1e6)
	globalFactor := mcts.randomness * float32(math.Sqrt(float64(cn.totalCount)+epsilon))
	for ii := range cn.actions {
		actionScore := (cn.Q[ii] + globalFactor*cn.actionsProbs[ii]/(1+float32(cn.count[ii])))
		if actionScore > bestActionScore {
			best = ii
			bestActionScore = actionScore
		}
	}
	return best
}

func (cn *cacheNode) FindBestScore(mcts *mctsSearcher) (bestIdx int, bestScore float32, actionsLabels []float32) {
	if len(cn.actions) <= 1 {
		// There is either one or none actions. Index will be set to 0 (first action)
		// or -1 (represents a SKIP_ACTION) respectively, and the score should be the
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
		glog.Errorf("MCTS.FindBestNode() called with no visits having been made.")
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
	for ii := 0; ii < len(cn.actions); ii++ {
		if cn.count[ii] > bestCount || (cn.count[ii] == bestCount && cn.Q[ii] > bestScore) {
			bestIdx = ii
			bestCount = cn.count[ii]
			bestScore = cn.Q[ii]
		}
	}
	bestScore = cn.sumMCScores[bestIdx] / float32(cn.count[bestIdx])
	return
}

func abs32(x float32) float32 { return float32(math.Abs(float64(x))) }

// Traverse traverses the game tree up to the given depth, and returns the
// expected score returned by the leaf node (or deepest node) visited.
// The score is with respect to the player playing (board.NextPlayer) at the node cn.
func (cn *cacheNode) Traverse(mcts *mctsSearcher, depthLeft int) float32 {
	// Checks end of game conditions.
	if cn.board.IsFinished() {
		_, score := ai.EndGameScore(cn.board)
		return score
	}

	// Checks max depth reached.
	if depthLeft == 0 {
		return cn.score
	}

	// Checks if score threshold is reached.
	if depthLeft < mcts.maxDepth-DEPTH_CHECK_MAX_ABS_SCORE {
		if abs32(cn.score) >= mcts.maxAbsScore {
			return cn.score
		}
	}

	// Sample new action.
	actionIdx := cn.Sample(mcts)
	nextCN := cn.Step(mcts, actionIdx)
	score := -nextCN.Traverse(mcts, depthLeft-1)

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
	// Scores of Q are normalized from +1 to -1
	cn.Q[actionIdx] = cn.sumMCScores[actionIdx] / (10 * float32(cn.count[actionIdx]))
	cn.mu.Unlock()

	return score
}

// runMCTS runs MCTS for the given specifications on the cacheNode.
func (mcts *mctsSearcher) runOnCN(cn *cacheNode) {
	// Sample while there is time.
	if len(cn.actions) > 1 {
		start := time.Now()
		count := 0

		// Handle parallelism.
		maxParallel := runtime.GOMAXPROCS(0)
		// TODO: parallelized forced to false for now (see NewMCSTSearcher). Still need to implement
		//   locks such that no traverse share paths. Also the algorithm should change since it changes the
		//   counts
		if !mcts.parallelized {
			maxParallel = 1
		}
		var wg sync.WaitGroup
		semaphore := make(chan bool, maxParallel)
		glog.V(3).Infof("MCTS: parallelization=%d", maxParallel)

		// Loop over traverses.
		for time.Since(start) < mcts.maxTime && count < mcts.maxTraverses {
			wg.Add(1)
			semaphore <- true
			go func() {
				glog.V(3).Infof("MCTS: starting traverse")
				cn.Traverse(mcts, mcts.maxDepth)
				glog.V(3).Infof("MCTS: done traverse")
				<-semaphore
				wg.Done()
			}()
			count++
		}
		wg.Wait()
		glog.V(1).Infof("MCTS traverses done: %d", count)
	}
}

func (mcts *mctsSearcher) measuredRunOnCN(cn *cacheNode) {
	beforeCacheNodes := mcts.numCacheNodes
	start := time.Now()
	mcts.runOnCN(cn)
	elapsedTime := time.Since(start)
	searchCacheNodes := mcts.numCacheNodes - beforeCacheNodes

	glog.V(1).Infof("States searched in this move:    \t%d CacheNodes",
		searchCacheNodes)
	cacheNodesPerSec := float64(searchCacheNodes) / float64(elapsedTime.Seconds())
	glog.V(1).Infof("Rate of evaluations in this move:\t%.1f CacheNodes/s",
		cacheNodesPerSec)

	glog.V(1).Infof("States searched in match so far: \t%d CacheNodes",
		mcts.numCacheNodes)
}

// Search implements the Searcher interface.
func (mcts *mctsSearcher) Search(b *Board) (
	action Action, board *Board, score float32, actionsLabels []float32) {
	cn := newCacheNode(mcts, b)
	if glog.V(1) {
		// Measure time and boards evaluated.
		mcts.measuredRunOnCN(cn)
	} else {
		mcts.runOnCN(cn)
	}

	var actionIdx int
	actionIdx, score, actionsLabels = cn.FindBestScore(mcts)
	if actionIdx >= 0 {
		action = cn.actions[actionIdx]
		board = cn.cacheNodes[actionIdx].board
	} else {
		action = SKIP_ACTION
		board = cn.cacheNodes[0].board
	}
	glog.V(1).Infof("Selected action %s, score %.2f", action, score)
	return
}

// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions)+1, and
// len(actionsLabels) == len(actions).
func (mcts *mctsSearcher) ScoreMatch(b *Board, actions []Action) (
	scores []float32, actionsLabels [][]float32) {
	var cn *cacheNode
	cn = newCacheNode(mcts, b)

	for _, action := range actions {
		// Search current node.
		if glog.V(1) {
			// Measure time and boards evaluated.
			mcts.measuredRunOnCN(cn)
		} else {
			mcts.runOnCN(cn)
		}

		// Score of this node, is the score of the best action.
		_, score, boardActionsLabels := cn.FindBestScore(mcts)
		scores = append(scores, score)
		actionsLabels = append(actionsLabels, boardActionsLabels)

		// The action taken may be different than the best action, specially
		// as the model evolves.
		var playedIdx int
		if action.IsSkipAction() {
			playedIdx = -1
		} else {
			playedIdx = cn.board.FindActionDeep(action)
		}
		cn = cn.Step(mcts, playedIdx)
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
