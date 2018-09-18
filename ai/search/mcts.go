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
	randomness float64
	priorBase  float32 // How much to weight the baseScore in comparison to MC samples.

	// Cache previous searches on the current tree. Reused by score match.
	reuseCN *cacheNode
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
	baseScores   []float32 // Scores for board.NextPlayer.
	count        []int     // How many times each of the paths have been traversed.
	sumMCScores  []float32 // Sum of each of the traversals at the given path.
	exponents    []float64
	sumExponents float64

	cacheNodes []*cacheNode
}

func newCacheNode(b *Board, scorer ai.BatchScorer, randomness float64) *cacheNode {
	cn := &cacheNode{board: b}
	cn.actions, cn.newBoards, cn.baseScores = ScoredActions(b, scorer)
	cn.count = make([]int, len(cn.actions))
	cn.sumMCScores = make([]float32, len(cn.actions))
	cn.exponents = make([]float64, len(cn.actions))
	for ii, score := range cn.baseScores {
		cn.exponents[ii] = math.Exp(float64(score) / randomness)
		cn.sumExponents += cn.exponents[ii]
	}
	cn.cacheNodes = make([]*cacheNode, len(cn.actions))
	return cn
}

// UpdateBaseScores re-scores the boards according to a presumably updates scorer.
// It is used by ScoreMatch with reuse=true.
func (cn *cacheNode) UpdateBaseScores(scorer ai.BatchScorer, randomness float64) {
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
		scored := scorer.BatchScore(boardsToScore)
		scoredIdx := 0
		for ii := range cn.baseScores {
			if !cn.newBoards[ii].IsFinished() {
				cn.baseScores[ii] = -scored[scoredIdx]
				scoredIdx++
			}
		}
	}

	// Finally update the exponents.
	cn.sumExponents = 0
	for ii, score := range cn.baseScores {
		cn.exponents[ii] = math.Exp(float64(score) / randomness)
		cn.sumExponents += cn.exponents[ii]
	}
}

// Step into the index's cacheNode under the current one. If it doesn't exist,
// create one using the given scorer.
func (cn *cacheNode) Step(
	index int, scorer ai.BatchScorer, randomness float64) *cacheNode {
	if cn.cacheNodes[index] == nil {
		cn.cacheNodes[index] = newCacheNode(cn.newBoards[index], scorer, randomness)
	}
	if cn.cacheNodes[index].baseScores == nil {
		cn.cacheNodes[index].UpdateBaseScores(scorer, randomness)
	}
	return cn.cacheNodes[index]
}

func (cn *cacheNode) ClearScores() {
	cn.baseScores = nil
	for ii := range cn.actions {
		cn.sumMCScores[ii] = 0
		cn.count[ii] = 0
		cn.exponents[ii] = 0
	}
	cn.sumExponents = 0

	for _, childCN := range cn.cacheNodes {
		if childCN != nil {
			childCN.ClearScores()
		}
	}
}

func (cn *cacheNode) Sample() int {
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
	if chance > 0.0001 {
		glog.Errorf("MCTS failed to choose any, %.3f probability mass still missing", chance)
	}
	return len(cn.exponents) - 1 // Pick last one.
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

func (cn *cacheNode) Traverse(
	depth int, scorer ai.BatchScorer, priorBase float32, randomness float64) float32 {
	// Sample according to current scores.
	ii := cn.Sample()
	if depth == 0 || cn.newBoards[ii].IsFinished() {
		// If leaf node, return base score.
		return cn.baseScores[ii]
	}

	// Traverse down the sampled variation.
	nextCN := cn.Step(ii, scorer, randomness)
	sampledScore := -nextCN.Traverse(depth-1, scorer, priorBase, randomness)

	// Propagate back the score.
	cn.sumMCScores[ii] += sampledScore
	cn.count[ii]++
	cn.sumExponents -= cn.exponents[ii]
	cn.exponents[ii] = math.Exp(float64(cn.EstimatedScore(ii, priorBase)) / randomness)
	cn.sumExponents += cn.exponents[ii]

	return sampledScore
}

// Search implements the Searcher interface.
func (mcts *mctsSearcher) Search(b *Board, scorer ai.BatchScorer) (
	action Action, board *Board, score float32) {
	cn := newCacheNode(b, scorer, mcts.randomness)
	mcts.runOnCN(cn, scorer)

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

// runMCTS runs MCTS for the given specifications on the cacheNode.
func (mcts *mctsSearcher) runOnCN(cn *cacheNode, scorer ai.BatchScorer) {
	// Sample while there is time.
	if len(cn.actions) > 1 {
		start := time.Now()
		count := 0
		for time.Since(start) < mcts.maxTime {
			cn.Traverse(mcts.maxDepth, scorer, mcts.priorBase, mcts.randomness)
			count++
		}
		glog.V(1).Infof("Samples: %d", count)
	}
}

// ScoreMatch will score the board at each board position, starting from the current one,
// and following each one of the actions. In the end, len(scores) == len(actions)+1.
// If useCache is true it will try to reuse previous iteration of boards generated,
// greatly accelerating things. But it has to be called with the same board.
func (mcts *mctsSearcher) ScoreMatch(
	b *Board, scorer ai.BatchScorer, actions []Action, reuse bool) (
	scores []float32) {
	var cn *cacheNode
	if reuse && mcts.reuseCN != nil {
		cn = mcts.reuseCN
		cn.ClearScores()
		cn.UpdateBaseScores(scorer, mcts.randomness)
	} else {
		cn = newCacheNode(b, scorer, mcts.randomness)
		if reuse {
			mcts.reuseCN = cn
		}
	}
	for _, action := range actions {
		mcts.runOnCN(cn, scorer)
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
		cn = cn.Step(idx, scorer, mcts.randomness)
	}

	// Add the final board score, if the match hasn't ended yet.
	_, score := cn.FindBestScore(mcts.priorBase)
	scores = append(scores, score)
	return
}
