// Package mcts is a Monte Carlo Tree Search implementation of searchers.Searcher for
// the Alpha-Zero algorithm.
//
// References used, since the original paper doesn't actually provide the formulas:
//
//   - https://suragnair.github.io/posts/alphazero.html by Surag Nair
//   - Paper here: https://github.com/suragnair/alpha-zero-general/blob/master/pretrained_models/writeup.pdf
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
	"math/rand"
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

type Searcher struct {
	// maxTime defines the maximum number of time to spend thinking.
	// Either maxTime or maxTraverses must be defined.
	maxTime time.Duration

	// maxTraverses, minTraverses define the limit number of traverses to do during the search, if not zero.
	// Either maxTime or maxTraverses must be defined.
	maxTraverses, minTraverses int
	maxAbsScore                float32 // Max absolute score, value above that interrupt the search.
	cPuct                      float32 // Degree of exploration of alpha-zero.

	// temperature (usually represented as the greek letter τ) is an exponent applied
	// to the counts used in the policy distribution (π) formula. If set to zero, it will
	// always take the best estimate action. AlphaZero Go uses 1 for the first 30 moves.
	// Larger models will make the play more random.
	temperature float32

	// maxRandDepth defines the move (in plies) after which temperature is disabled and
	// it simply takes the best move, as opposed to randomly using th policy distribution.
	// A value <= 0 means there is no maxRandDepth.
	maxRandDepth int

	// If to explore paths in parallel. Not yet supported.
	parallelized bool

	// ValueScorer to use during search.
	scorer ai.PolicyScorer
}

type matchStats struct {
	// numEvaluations of board positions, should be equal to the number of traverses.
	numEvaluations int
	// numCacheNodes created.
	numCacheNodes int
}

// Clone searcher.
func (s *Searcher) Clone() *Searcher {
	newMCTS := &Searcher{}
	*newMCTS = *s
	return newMCTS
}

func (s *Searcher) WithScorer(scorer ai.PolicyScorer) *Searcher {
	s.scorer = scorer
	return s
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
func (s *Searcher) newCacheNode(b *Board, stats *matchStats) (*cacheNode, error) {
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
	cn.actionsProbs = s.scorer.PolicyScore(b)

	// Sanity check:
	var sumProbs float32
	for _, prob := range cn.actionsProbs {
		if prob < 0 {
			klog.Errorf("Board has negative action probability %g !?", prob)
			ui := cli.New(true, false)
			fmt.Println()
			ui.PrintBoard(cn.board)
			fmt.Printf("Available actions: %v\n", cn.board.Derived.Actions)
			fmt.Printf("Probabilities: %v\n", cn.actionsProbs)
			return nil, errors.Errorf("board scorer returned negative probability %g for board position", prob)
		}
		sumProbs += prob
	}
	if math.Abs(float64(sumProbs-1.0)) > 1e-3 {
		ui := cli.New(true, false)
		fmt.Println()
		ui.PrintBoard(cn.board)
		fmt.Printf("Available actions: %v\n", cn.board.Derived.Actions)
		fmt.Printf("Probabilities: %v\n", cn.actionsProbs)
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
func (s *Searcher) SearchSubtree(cn *cacheNode, stats *matchStats) (score float32, err error) {
	// Find the action with the best upper confidence (U in the description).
	bestAction := -1
	bestUpperConfidence := float32(math.Inf(-1))
	globalFactor := s.cPuct * float32(math.Sqrt(float64(cn.sumN)))
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
			score = -s.scorer.Score(newBoard)
		}
		cn.N[bestAction] = 1
		cn.sumN++
		cn.sumScores[bestAction] += score
		stats.numEvaluations++
		return
	}

	// If not the first time we sample the action, make sure we have a corresponding
	// cacheNode for it, expanding the tree.
	if cn.cacheNodes[bestAction] == nil {
		newBoard := cn.board.TakeAllActions()[bestAction]
		if isEnd, endScore := ai.IsEndGameAndScore(newBoard); isEnd {
			// Return immediately and don't create a cacheNode.
			score = -endScore
			cn.N[bestAction]++
			cn.sumN++
			cn.sumScores[bestAction] += score
			stats.numEvaluations++
			return
		}

		cn.cacheNodes[bestAction], err = s.newCacheNode(newBoard, stats)
		if err != nil {
			return
		}
	}

	// Recursively sample value of the best action.
	score, err = s.SearchSubtree(cn.cacheNodes[bestAction], stats)
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
func (s *Searcher) Search(board *Board) (action Action, nextBoard *Board, score float32, err error) {
	action, nextBoard, score, _, err = s.searchImpl(board, false)
	return
}

// SearchWithPolicy search and returns the policy (actionsProbabilities) derived from the search.
func (s *Searcher) SearchWithPolicy(board *Board) (action Action, nextBoard *Board, score float32, policy []float32, err error) {
	return s.searchImpl(board, true)
}

func (s *Searcher) searchImpl(board *Board, withPolicy bool) (action Action, nextBoard *Board, score float32, policy []float32, err error) {
	var rootCacheNode *cacheNode
	var stats matchStats
	rootCacheNode, err = s.newCacheNode(board, &stats)
	if err != nil {
		return
	}

	// Keep sampling until the time is over.
	numTraverses := 0
	startTime := time.Now()
	var elapsed time.Duration
	for {
		_, err = s.SearchSubtree(rootCacheNode, &stats)
		if err != nil {
			return
		}
		numTraverses++

		if s.maxTraverses > 0 && numTraverses >= s.maxTraverses {
			break
		}
		if s.minTraverses > 0 && numTraverses < s.minTraverses {
			continue
		}
		if s.maxTime > 0 {
			elapsed = time.Since(startTime)
			if elapsed > s.maxTime {
				break
			}
		}
	}

	// Log performance.
	if klog.V(1).Enabled() {
		elapsed = time.Since(startTime)
		evaluationsRate := float64(stats.numEvaluations) / elapsed.Seconds()
		klog.Infof("Move #%d MCTS searched %d nodes at %.2f nodes/s", board.MoveNumber, stats.numEvaluations, evaluationsRate)
	}

	bestActionIdx := s.selectAction(rootCacheNode)
	action = board.Derived.Actions[bestActionIdx]
	nextBoard = board.TakeAllActions()[bestActionIdx]
	score = rootCacheNode.sumScores[bestActionIdx] / float32(rootCacheNode.N[bestActionIdx])
	if withPolicy {
		policy = s.derivedPolicy(rootCacheNode)
	}
	return
}

func pow32(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

// selectAction given the root of the MCTS expanded search.
// If temperature is 0, or maxRandDepth is reached, it is greedy.
// Otherwise, it picks randomly from a probability distribution based on the number of visits of
// each sub-tree.
func (s *Searcher) selectAction(rootCacheNode *cacheNode) int {
	board := rootCacheNode.board
	if s.temperature == 0 || (s.maxRandDepth > 0 && board.MoveNumber > s.maxRandDepth) {
		// Greedily pick best action and its estimate.
		bestActionIdx, mostVisits := -1, -1
		for actionIdx, nVisits := range rootCacheNode.N {
			if nVisits > mostVisits {
				mostVisits = nVisits
				bestActionIdx = actionIdx
			}
		}
		return bestActionIdx
	}

	// Calculate policy probability distribution based on visits (not the one returned by the model)
	numActions := len(rootCacheNode.N)
	actionsProbs := make([]float32, numActions)
	temp := s.temperature
	for actionIdx, nVisits := range rootCacheNode.N {
		actionsProbs[actionIdx] = float32(nVisits) / float32(rootCacheNode.sumN)
		if temp != 1 {
			actionsProbs[actionIdx] = pow32(actionsProbs[actionIdx], 1/temp)
		}
	}
	// Normalize probabilities
	if temp != 1 {
		sumProbs := float32(0)
		for _, prob := range actionsProbs {
			sumProbs += prob
		}
		for actionIdx, prob := range actionsProbs {
			actionsProbs[actionIdx] = prob / sumProbs
		}
	}
	// Pick random action from probability distribution.
	r := rand.Float32()
	sumProb := float32(0.0)
	for actionIdx, prob := range actionsProbs {
		sumProb += prob
		if r <= sumProb {
			return actionIdx
		}
	}
	// Due to rounding errors we may get here, in this case return last action.
	return len(actionsProbs) - 1
}

// derivedPolicy returns the policy used for learning, based on root cacheNode.
func (s *Searcher) derivedPolicy(rootCacheNode *cacheNode) []float32 {
	numActions := len(rootCacheNode.N)
	actionsProbs := make([]float32, numActions)
	for actionIdx, nVisits := range rootCacheNode.N {
		actionsProbs[actionIdx] = float32(nVisits) / float32(rootCacheNode.sumN)
	}
	return actionsProbs
}
