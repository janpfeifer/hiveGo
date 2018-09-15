package search

// Monte Carlo Tree Search implementation

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/janpfeifer/hiveGo/ai"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf
var _ = fmt.Printf

type mctsSearcher struct {
	maxDepth   int
	maxTime    time.Duration
	randomness float64
	priorBase  float64 // How much to weight the baseScore in comparison to MC samples.
}

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPrunning.
func NewMonteCarloTreeSearcher(maxDepth int, maxTime time.Duration, randomness float64) Searcher {
	return &mctsSearcher{maxDepth: maxDepth, maxTime: maxTime, randomness: randomness, priorBase: 3.0}
}

// cacheNode holds information about the possible actions of a board.
type cacheNode struct {
	board        *Board
	actions      []Action
	newBoards    []*Board
	baseScores   []float64 // Scores for board.NextPlayer.
	count        []int     // How many times each of the paths have been traversed.
	sumMCScores  []float64 // Sum of each of the traversals at the given path.
	exponents    []float64
	sumExponents float64

	cacheNodes []*cacheNode
}

func newCacheNode(b *Board, scorer ai.BatchScorer) *cacheNode {
	cn := &cacheNode{board: b}
	cn.actions, cn.newBoards, cn.baseScores = ScoredActions(b, scorer)
	cn.count = make([]int, len(cn.actions))
	cn.sumMCScores = make([]float64, len(cn.actions))
	cn.exponents = make([]float64, len(cn.actions))
	for ii, score := range cn.baseScores {
		cn.exponents[ii] = math.Exp(score)
		cn.sumExponents += cn.exponents[ii]
	}
	cn.cacheNodes = make([]*cacheNode, len(cn.actions))
	return cn
}

func (cn *cacheNode) Sample() int {
	chance := rand.Float64()
	for ii, exponent := range cn.exponents {
		probability := exponent / cn.sumExponents
		if chance <= probability {
			return ii
		}
		chance -= probability
	}
	log.Printf("MCTS failed to choose any, %.3f probability mass still missing", chance)
	return len(cn.exponents) - 1
}

func (cn *cacheNode) EstimatedScore(idx int, priorBase float64) float64 {
	estimatedScore := cn.baseScores[idx]*priorBase + cn.sumMCScores[idx]
	estimatedScore /= priorBase + float64(cn.count[idx])
	if estimatedScore > 10.0 {
		estimatedScore = 10.0
	} else if estimatedScore < -10.0 {
		estimatedScore = -10.0
	}
	return estimatedScore
}

func (cn *cacheNode) Traverse(depth int, scorer ai.BatchScorer, priorBase float64) float64 {
	// Sample according to current scores.
	ii := cn.Sample()
	if depth == 0 {
		// If leaf node, return base score.
		return cn.baseScores[ii]
	}

	// Traverse down the sampled variation.
	if cn.cacheNodes[ii] == nil {
		cn.cacheNodes[ii] = newCacheNode(cn.newBoards[ii], scorer)
	}
	sampledScore := -cn.cacheNodes[ii].Traverse(depth-1, scorer, priorBase)

	// Propagate back the score.
	cn.sumMCScores[ii] += sampledScore
	cn.count[ii]++
	cn.sumExponents -= cn.exponents[ii]
	cn.exponents[ii] = math.Exp(cn.EstimatedScore(ii, priorBase))
	cn.sumExponents += cn.exponents[ii]

	return sampledScore
}

// Search implements the Searcher interface.
func (mcts *mctsSearcher) Search(b *Board, scorer ai.BatchScorer) (
	action Action, board *Board, score float64) {
	cn := newCacheNode(b, scorer)

	// Sample while there is time.
	if len(cn.actions) > 1 {
		start := time.Now()
		count := 0
		for time.Since(start) < mcts.maxTime {
			cn.Traverse(mcts.maxDepth, scorer, mcts.priorBase)
			count++
		}
		log.Printf("Samples: %d", count)
	}

	if false {
		for ii, action := range cn.actions {
			mean := cn.sumMCScores[ii]
			if cn.count[ii] > 0 {
				mean /= float64(cn.count[ii])
			}
			log.Printf("Action %s:\tbase=%.2f\testimated=%.2f\tmean=%.2f\tprob=%.2f%%\tcount=%d",
				action, cn.baseScores[ii], cn.EstimatedScore(ii, mcts.priorBase),
				mean, 100.0*cn.exponents[ii]/cn.sumExponents, cn.count[ii])
		}
		log.Printf("")
	}

	// Select best action.
	bestIdx := 0
	bestScore := cn.EstimatedScore(0, mcts.priorBase)
	for ii := 1; ii < len(cn.actions); ii++ {
		score := cn.EstimatedScore(ii, mcts.priorBase)
		if score > bestScore {
			bestScore = score
			bestIdx = ii
		}
	}

	log.Printf("Estimated best score: %.2f", bestScore)
	return cn.actions[bestIdx], cn.newBoards[bestIdx], bestScore
}