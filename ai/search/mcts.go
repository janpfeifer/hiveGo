package search

// Monte Carlo Tree Search implementation

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf
var _ = fmt.Printf

type mctsSearcher struct {
	maxDepth   int
	maxTime    time.Duration
	randomness float32
	priorBase  float32 // How much to weight the baseScore in comparison to MC samples.
}

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPrunning.
func NewMonteCarloTreeSearcher(maxDepth int, maxTime time.Duration, randomness float32) Searcher {
	return &mctsSearcher{maxDepth: maxDepth, maxTime: maxTime, randomness: randomness, priorBase: 3.0}
}

// cacheNode holds information about the possible actions of a board.
type cacheNode struct {
	board        *Board
	actions      []Action
	newBoards    []*Board
	baseScores   []float32 // Scores for board.NextPlayer.
	count        []int     // How many times each of the paths have been traversed.
	sumMCScores  []float32 // Sum of each of the traversals at the given path.
	exponents    []float32
	sumExponents float32

	cacheNodes []*cacheNode
}

func newCacheNode(b *Board, scorer ai.BatchScorer) *cacheNode {
	cn := &cacheNode{board: b}
	cn.actions, cn.newBoards, cn.baseScores = ScoredActions(b, scorer)
	cn.count = make([]int, len(cn.actions))
	cn.sumMCScores = make([]float32, len(cn.actions))
	cn.exponents = make([]float32, len(cn.actions))
	for ii, score := range cn.baseScores {
		cn.exponents[ii] = float32(math.Exp(float64(score)))
		cn.sumExponents += cn.exponents[ii]
	}
	cn.cacheNodes = make([]*cacheNode, len(cn.actions))
	return cn
}

func (cn *cacheNode) Sample() int {
	if len(cn.actions) == 1 {
		return 0
	}
	chance := rand.Float64()
	for ii, exponent := range cn.exponents {
		probability := float64(exponent / cn.sumExponents)
		if chance <= probability {
			return ii
		}
		chance -= probability
	}
	glog.Errorf("MCTS failed to choose any, %.3f probability mass still missing", chance)
	return len(cn.exponents) - 1
}

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

func (cn *cacheNode) FindBestScore(priorBase float32) (bestIdx int, bestScore float32) {
	// Select best action.
	bestIdx = 0
	bestScore = cn.EstimatedScore(0, priorBase)
	for ii := 1; ii < len(cn.actions); ii++ {
		score := cn.EstimatedScore(ii, priorBase)
		if score > bestScore {
			bestScore = score
			bestIdx = ii
		}
	}
	return
}

func (cn *cacheNode) EstimatedScore(idx int, priorBase float32) float32 {
	estimatedScore := cn.baseScores[idx]*priorBase + cn.sumMCScores[idx]
	estimatedScore /= priorBase + float32(cn.count[idx])
	if estimatedScore > 10.0 {
		estimatedScore = 10.0
	} else if estimatedScore < -10.0 {
		estimatedScore = -10.0
	}
	return estimatedScore
}

func (cn *cacheNode) Traverse(depth int, scorer ai.BatchScorer, priorBase float32) float32 {
	// Sample according to current scores.
	ii := cn.Sample()
	if depth == 0 || cn.newBoards[ii].IsFinished() {
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
	cn.exponents[ii] = float32(math.Exp(float64(cn.EstimatedScore(ii, priorBase))))
	cn.sumExponents += cn.exponents[ii]

	return sampledScore
}

// Search implements the Searcher interface.
func (mcts *mctsSearcher) Search(b *Board, scorer ai.BatchScorer) (
	action Action, board *Board, score float32) {
	cn := newCacheNode(b, scorer)
	mcts.updateCN(cn, scorer)

	if glog.V(2) {
		for ii, action := range cn.actions {
			mean := cn.sumMCScores[ii]
			if cn.count[ii] > 0 {
				mean /= float32(cn.count[ii])
			}
			glog.Infof("Action %s:\tbase=%.2f\testimated=%.2f\tmean=%.2f\tprob=%.2f%%\tcount=%d",
				action, cn.baseScores[ii], cn.EstimatedScore(ii, mcts.priorBase),
				mean, 100.0*cn.exponents[ii]/cn.sumExponents, cn.count[ii])
		}
		glog.Infoln("")
	}
	bestIdx, bestScore := cn.FindBestScore(mcts.priorBase)

	glog.V(1).Infof("Estimated best score: %.2f", bestScore)
	return cn.actions[bestIdx], cn.newBoards[bestIdx], bestScore
}

// updateCN runs MCTS for the given specifications on the cacheNode.
func (mcts *mctsSearcher) updateCN(cn *cacheNode, scorer ai.BatchScorer) {
	// Sample while there is time.
	if len(cn.actions) > 1 {
		start := time.Now()
		count := 0
		for time.Since(start) < mcts.maxTime {
			cn.Traverse(mcts.maxDepth, scorer, mcts.priorBase)
			count++
		}
		glog.V(1).Infof("Samples: %d", count)
	}
}

// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions)+1.
func (mcts *mctsSearcher) ScoreMatch(b *Board, scorer ai.BatchScorer, actions []Action) (
	scores []float32) {
	cn := newCacheNode(b, scorer)
	for _, action := range actions {
		mcts.updateCN(cn, scorer)
		// Score of this node, is the score of the best action.
		_, score := cn.FindBestScore(mcts.priorBase)
		scores = append(scores, score)

		// The action taken may be different than the best action, specially
		// as the model evolves.
		idx := cn.FindAction(action)
		if isEnd, score := ai.EndGameScore(cn.newBoards[idx]); isEnd {
			scores = append(scores, score)
			return
		}
		newCn := cn.cacheNodes[idx]
		if newCn == nil {
			newCn = newCacheNode(cn.newBoards[idx], scorer)
		}
		cn = newCn
	}

	// Add the final board score, if the match hasn't ended yet.
	_, score := cn.FindBestScore(mcts.priorBase)
	scores = append(scores, score)
	return
}
