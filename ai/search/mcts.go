package search

// Monte Carlo Tree Search implementation, with a hacked version of UCT
// (Upper Confidence Bound for Trees).

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf
var _ = fmt.Printf

type mctsSearcher struct {
	maxDepth     int
	maxTime      time.Duration
	maxTraverses int
	useUCT       bool

	// Number of boards and candidate nodes generated during a
	// search: used for performance measures.
	numBoards     int
	numCacheNodes int

	// For UCT, scores higher than that will stop the traverse, the victory
	// or defeat is assumed a given.
	maxScore float32

	scorer       ai.BatchScorer
	randomness   float64
	priorBase    float32 // How much to weight the baseScore in comparison to MC samples.
	parallelized bool

	// Cache previous searches on the current tree. Reused by score match.
	reuseCN *cacheNode
}

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPrunning.
func NewMonteCarloTreeSearcher(maxDepth int, maxTime time.Duration,
	maxTraverses int, useUCT bool, maxScore float32,
	scorer ai.BatchScorer, randomness float64, parallelized bool) Searcher {
	return &mctsSearcher{
		maxDepth:     maxDepth,
		maxTime:      maxTime,
		maxTraverses: maxTraverses,
		useUCT:       useUCT,
		maxScore:     maxScore,

		numBoards:     0,
		numCacheNodes: 0,

		scorer:       scorer,
		randomness:   randomness,
		priorBase:    3.0,
		parallelized: parallelized,
	}
}

// cacheNode holds information about the possible actions of a board.
type cacheNode struct {
	// Board, actions, children boards and children base scores.
	board      *Board
	actions    []Action
	newBoards  []*Board
	baseScores []float32 // Scores for board.NextPlayer.

	// Children cacheNodes.
	cacheNodes   []*cacheNode
	cacheNodesMu []sync.Mutex

	// How many times each of the paths have been traversed.
	count []int

	// Random MCTS.
	sumMCScores  []float32 // Sum of each of the traversals at the given path.
	exponents    []float64
	sumExponents float64

	// UCT UpperBound based MCTS.
	mctsScores  []float32
	upperBounds []float32

	mu sync.Mutex // Lcok for updates.
}

func newCacheNode(mcts *mctsSearcher, b *Board) *cacheNode {
	cn := &cacheNode{board: b}
	mcts.numCacheNodes++
	cn.actions, cn.newBoards, cn.baseScores = ScoredActions(b, mcts.scorer)
	mcts.numBoards += len(cn.newBoards)
	cn.count = make([]int, len(cn.actions))
	if mcts.useUCT {
		cn.mctsScores = make([]float32, len(cn.actions))
		cn.upperBounds = make([]float32, len(cn.actions))
	} else {
		cn.sumMCScores = make([]float32, len(cn.actions))
		cn.exponents = make([]float64, len(cn.actions))
		cn.SetSoftmaxScoresFromBase(mcts)
	}
	cn.cacheNodes = make([]*cacheNode, len(cn.actions))
	cn.cacheNodesMu = make([]sync.Mutex, len(cn.actions))
	return cn
}

func (cn *cacheNode) SetSoftmaxScoresFromBase(mcts *mctsSearcher) {
	cn.sumExponents = 0
	for ii, score := range cn.baseScores {
		cn.exponents[ii] = math.Exp(float64(score) / mcts.randomness)
		cn.sumExponents += cn.exponents[ii]
	}
}

// UpdateBaseScores re-scores the boards according to a presumably
// updated scorer. It is used by ScoreMatch with rescore=true.
func (cn *cacheNode) UpdateBaseScores(mcts *mctsSearcher) {
	cn.baseScores = make([]float32, len(cn.actions))
	boardsToScore := make([]*Board, 0, len(cn.newBoards))

	// First score end-of-game boards.
	for ii, board := range cn.newBoards {
		if isEnd, score := ai.EndGameScore(board); isEnd {
			// End game is treated differently.
			cn.baseScores[ii] = -score
		} else {
			boardsToScore = append(boardsToScore, board)
		}
	}

	// Score non-end-of-game, using scorer.
	if len(boardsToScore) > 0 {
		// Score other boards.
		scored, _ := mcts.scorer.BatchScore(boardsToScore)
		scoredIdx := 0
		for ii := range cn.baseScores {
			if !cn.newBoards[ii].IsFinished() {
				cn.baseScores[ii] = -scored[scoredIdx]
				scoredIdx++
			}
		}
	}

	// Finally update the exponents.
	if !mcts.useUCT {
		cn.SetSoftmaxScoresFromBase(mcts)
	}
}

// Step steps into the index's cacheNode under the current one. If it doesn't exist,
// it creates a new one.
func (cn *cacheNode) Step(mcts *mctsSearcher, index int) *cacheNode {
	cn.cacheNodesMu[index].Lock()
	defer cn.cacheNodesMu[index].Unlock()
	if cn.cacheNodes[index] == nil {
		cn.cacheNodes[index] = newCacheNode(mcts, cn.newBoards[index])
	} else if cn.cacheNodes[index].baseScores == nil {
		// If this is rescoring a previously already played match,
		// the only thing missing will be rescorign the baseScores.
		cn.cacheNodes[index].UpdateBaseScores(mcts)
	}
	return cn.cacheNodes[index]
}

func (cn *cacheNode) RecursivelyClearScores(mcts *mctsSearcher) {
	cn.baseScores = nil
	if mcts.useUCT {
		for ii := range cn.actions {
			cn.count[ii] = 0
			cn.mctsScores[ii] = 0
		}
	} else {
		for ii := range cn.actions {
			cn.count[ii] = 0
			cn.sumMCScores[ii] = 0
			cn.exponents[ii] = 0
		}
		cn.sumExponents = 0
	}

	for _, childCN := range cn.cacheNodes {
		if childCN != nil {
			childCN.RecursivelyClearScores(mcts)
		}
	}
}

// Sample picks the next step, with probability weighted
// by expected score. Not used if useUCT is set.
func (cn *cacheNode) Sample(mcts *mctsSearcher) int {
	cn.mu.Lock()
	defer cn.mu.Unlock()

	if len(cn.actions) == 0 {
		log.Panic("Sampling from no actions available.")
	}
	if len(cn.actions) == 1 {
		return 0
	}
	chance := rand.Float64()
	for ii, exponent := range cn.exponents {
		probability := exponent / cn.sumExponents
		if chance <= probability {
			return ii
		}
		chance -= probability
	}
	if chance > 0.001 {
		glog.Errorf("MCTS failed to choose any, %.3f probability mass still missing", chance)
	}
	return len(cn.exponents) - 1 // Pick last one.
}

// FindAction returns the index to the given action in the board.
func (cn *cacheNode) FindAction(action Action) int {
	for ii, cnAction := range cn.actions {
		if cnAction == action {
			return ii
		}
	}
	ui := ascii_ui.NewUI(true, false)
	ui.PrintBoard(cn.board)
	log.Panicf("Action %v chosen is not valid. Available: %v", action, cn.actions)
	return -1
}

func (cn *cacheNode) FindBestScore(mcts *mctsSearcher) (bestIdx int, bestScore float32) {
	// Select best action.
	bestIdx = 0
	bestScore = cn.EstimatedScore(mcts, 0)
	if len(cn.actions) == 0 {
		// TODO: fix this ... cn.EstimatedScore(mcts, 0) wouldn't work.
		return -1, bestScore
	}
	for ii := 1; ii < len(cn.actions); ii++ {
		score := cn.EstimatedScore(mcts, ii)
		if score > bestScore {
			bestScore = score
			bestIdx = ii
		}
	}
	return
}

// EstimatedScore returns current estimate of score for index action of the current node.
func (cn *cacheNode) EstimatedScore(mcts *mctsSearcher, idx int) float32 {
	if mcts.useUCT {
		// Score is the base score if no traverse was done. Otherwise takes
		// the current traverse score.
		if cn.count[idx] == 0 {
			return cn.baseScores[idx]
		} else {
			return cn.mctsScores[idx]
		}
	} else {
		if idx > len(cn.actions) || idx > len(cn.baseScores) {
			log.Panicf("Invalid index for EstimatedScore: %d, actions=%d, baseScores=%d, cn.sumMCScores=%d",
				idx, len(cn.actions), len(cn.baseScores), len(cn.sumMCScores))
		}
		estimatedScore := cn.baseScores[idx]*mcts.priorBase + cn.sumMCScores[idx]
		estimatedScore /= mcts.priorBase + float32(cn.count[idx])
		if estimatedScore > 10.0 {
			estimatedScore = 10.0
		} else if estimatedScore < -10.0 {
			estimatedScore = -10.0
		}
		return estimatedScore
	}
}

// FindHighestUpperBound finds the index of the highest estimated upper-bound
func (cn *cacheNode) FindHighestUpperBound(mcts *mctsSearcher) (highestUpperBoundIdx int) {
	// Select best action.
	upperBoundScores := cn.UpperBoundScores(mcts)
	highestUpperBoundIdx = 0
	highestUpperBound := upperBoundScores[0]
	for ii := 1; ii < len(cn.actions); ii++ {
		if upperBoundScores[ii] > highestUpperBound {
			highestUpperBound = upperBoundScores[ii]
			highestUpperBoundIdx = ii
		}
	}
	return
}

// UpperBoundScores returns the estimated upper-bound scores for the available actions.
// This is part of the UCT MCTS algorithm.
func (cn *cacheNode) UpperBoundScores(mcts *mctsSearcher) (scores []float32) {
	scores = make([]float32, len(cn.actions))
	totalCount := len(cn.actions) // Starts with one per action.
	if totalCount == 1 {
		scores[0] = cn.EstimatedScore(mcts, 0)
		return
	}
	for _, v := range cn.count {
		totalCount += v
	}
	lnTotalCount := math.Log(float64(totalCount))
	baseDeviation := float32(mcts.randomness * 5.)
	for ii := range scores {
		scores[ii] = cn.EstimatedScore(mcts, ii) + baseDeviation*float32(math.Sqrt(lnTotalCount/(float64(cn.count[ii])+1)))
	}
	return
}

// Traverse traverses the game tree up to the given depth, and returns the
// sampled (random MCTS) or expected (UCT) score.
func (cn *cacheNode) Traverse(mcts *mctsSearcher, depth int, depthThresholded int) float32 {
	if cn.baseScores == nil {
		log.Panic("cn::Traverse(): cacheNode has no baseScores")
	}

	// Sample/select according to current scores.
	var ii int
	if mcts.useUCT {
		// Threshold at given max score.
		if depth <= depthThresholded {
			_, score := cn.FindBestScore(mcts)
			if score > mcts.maxScore {
				return score
			} else if score < -mcts.maxScore {
				return score
			}
		}
		ii = cn.FindHighestUpperBound(mcts)
	} else {
		ii = cn.Sample(mcts)
	}
	if depth == 0 || cn.newBoards[ii].IsFinished() {
		// If leaf node, return base score.
		return cn.baseScores[ii]
	}

	// Traverse down the sampled variation.
	nextCN := cn.Step(mcts, ii)
	score := -nextCN.Traverse(mcts, depth-1, depthThresholded)

	// Propagate back the score.
	cn.mu.Lock()
	cn.count[ii]++
	if mcts.useUCT {
		// If using UCT we only store the most likely score, and
		// then return the best score so far.
		cn.mctsScores[ii] = score
		_, score = cn.FindBestScore(mcts)

	} else {
		// Random sampling.
		cn.sumMCScores[ii] += score
		cn.sumExponents -= cn.exponents[ii]
		cn.exponents[ii] = math.Exp(float64(cn.EstimatedScore(mcts, ii)) / mcts.randomness)
		cn.sumExponents += cn.exponents[ii]
	}
	cn.mu.Unlock()

	return score
}

func (mcts *mctsSearcher) measuredRunOnCN(cn *cacheNode) {
	start := time.Now()
	beforeCacheNodes := mcts.numCacheNodes
	beforeBoards := mcts.numBoards

	mcts.runOnCN(cn)

	searchCacheNodes := mcts.numCacheNodes - beforeCacheNodes
	searchBoards := mcts.numBoards - beforeBoards
	glog.V(1).Infof("States serached in this move:    \t%d CacheNodes,  \t%d Boards",
		searchCacheNodes, searchBoards)

	elapsedTime := time.Since(start)
	cacheNodesPerSec := float64(searchCacheNodes) / float64(elapsedTime.Seconds())
	boardsPerSec := float64(searchBoards) / float64(elapsedTime.Seconds())
	glog.V(1).Infof("Rate of evaluations in this move:\t%.1f CacheNodes/s,\t%.1f Boards/s",
		cacheNodesPerSec, boardsPerSec)

	glog.V(1).Infof("States serached in match so far: \t%d CacheNodes,  \t%d Boards",
		mcts.numCacheNodes, mcts.numBoards)
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

	if glog.V(2) {
		if mcts.useUCT {
			ubScores := cn.UpperBoundScores(mcts)
			for ii, action := range cn.actions {
				glog.Infof("Action %s:\tbase=%.2f\testimated=%.2f\tuppder-bound=%.2f\tcount=%d",
					action, cn.baseScores[ii], cn.EstimatedScore(mcts, ii),
					ubScores[ii], cn.count[ii])
			}
			glog.Infoln("")
		} else {
			for ii, action := range cn.actions {
				mean := cn.sumMCScores[ii]
				if cn.count[ii] > 0 {
					mean /= float32(cn.count[ii])
				}
				glog.Infof("Action %s:\tbase=%.2f\testimated=%.2f\tmean=%.2f\tprob=%.2f%%\tcount=%d",
					action, cn.baseScores[ii], cn.EstimatedScore(mcts, ii),
					mean, 100.0*cn.exponents[ii]/cn.sumExponents, cn.count[ii])
			}
			glog.Infoln("")
		}
	}

	var actionIdx int
	var actionScore float32
	if mcts.useUCT {
		actionIdx, actionScore = cn.FindBestScore(mcts)
		glog.V(1).Infof("Estimated best score: %.2f", actionScore)
	} else {
		actionIdx = cn.Sample(mcts)
		actionScore = cn.EstimatedScore(mcts, actionIdx)
		glog.V(1).Infof("Estimated action score: %.2f", actionScore)
	}

	// TODO: calculate actionsLabels
	return cn.actions[actionIdx], cn.newBoards[actionIdx], actionScore, nil
}

// runMCTS runs MCTS for the given specifications on the cacheNode.
func (mcts *mctsSearcher) runOnCN(cn *cacheNode) {
	// Sample while there is time.
	if len(cn.actions) > 1 {
		start := time.Now()
		count := 0

		// Handle parallelism.
		maxParallel := runtime.GOMAXPROCS(0)
		if !mcts.parallelized || mcts.useUCT {
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
				cn.Traverse(mcts, mcts.maxDepth, mcts.maxDepth-4)
				glog.V(3).Infof("MCTS: done traverse")
				<-semaphore
				wg.Done()
			}()
			count++
		}
		wg.Wait()
		glog.V(1).Infof("Samples: %d", count)
	}
}

// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions)+1.
func (mcts *mctsSearcher) ScoreMatch(b *Board, actions []Action) (
	scores []float32, actionsLabels [][]float32) {
	var cn *cacheNode
	cn = newCacheNode(mcts, b)

	for _, action := range actions {
		mcts.runOnCN(cn)
		// Score of this node, is the score of the best action.
		bestActionIdx, score := cn.FindBestScore(mcts)
		if len(cn.actions) > 0 {
			bestActionVec := make([]float32, len(b.Derived.Actions))
			bestActionVec[bestActionIdx] = 1
			actionsLabels = append(actionsLabels, bestActionVec)
		} else {
			actionsLabels = append(actionsLabels, nil)
		}
		scores = append(scores, score)

		// The action taken may be different than the best action, specially
		// as the model evolves.
		idx := cn.FindAction(action)
		if isEnd, score := ai.EndGameScore(cn.newBoards[idx]); isEnd {
			scores = append(scores, score)
			return
		}
		cn = cn.Step(mcts, idx)
	}

	// Add the final board score, if the match hasn't ended yet.
	_, score := cn.FindBestScore(mcts)
	scores = append(scores, score)
	return
}
