package search

// Monte Carlo Tree Search implementation

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
	randomness   float64
	priorBase    float32 // How much to weight the baseScore in comparison to MC samples.
	parallelized bool

	// Cache previous searches on the current tree. Reused by score match.
	reuseCN *cacheNode
}

// NewAlphaBetaSearcher returns a Searcher that implements AlphaBetaPrunning.
func NewMonteCarloTreeSearcher(maxDepth int, maxTime time.Duration, randomness float64, parallelized bool) Searcher {
	return &mctsSearcher{maxDepth: maxDepth, maxTime: maxTime, randomness: randomness, priorBase: 3.0, parallelized: parallelized}
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

	mu sync.Mutex // Lcok for updates.
}

var testCN *cacheNode

func newCacheNode(b *Board, scorer ai.BatchScorer, randomness float64) *cacheNode {
	if testCN != nil && testCN.baseScores == nil {
		log.Panic("cn::newCacheNode(): cacheNode (0) has no baseScores")
	}
	cn := &cacheNode{board: b}
	cn.actions, cn.newBoards, cn.baseScores = ScoredActions(b, scorer)
	if testCN != nil && testCN.baseScores == nil {
		log.Panic("cn::newCacheNode(): cacheNode (1) has no baseScores")
	}
	cn.count = make([]int, len(cn.actions))
	cn.sumMCScores = make([]float32, len(cn.actions))
	cn.exponents = make([]float64, len(cn.actions))
	for ii, score := range cn.baseScores {
		cn.exponents[ii] = math.Exp(float64(score) / randomness)
		cn.sumExponents += cn.exponents[ii]
	}
	cn.cacheNodes = make([]*cacheNode, len(cn.actions))
	if testCN != nil && testCN.baseScores == nil {
		log.Panic("cn::newCacheNode(): cacheNode (2) has no baseScores")
	}
	return cn
}

// UpdateBaseScores re-scores the boards according to a presumably updates scorer.
// It is used by ScoreMatch with rescore=true.
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
		cn.mu.Lock()
		if cn.cacheNodes[index] == nil {
			if cn.baseScores == nil {
				log.Panic("cn::Step(): cacheNode (2.a) has no baseScores")
			}
			testCN = cn
			cn.cacheNodes[index] = newCacheNode(cn.newBoards[index], scorer, randomness)
			if cn.baseScores == nil {
				log.Panicf("cn::Step(): cacheNode (2.b) has no baseScores, idx=%d, act=%d",
					index, len(cn.actions))
			}
		}
		cn.mu.Unlock()

	} else if cn.cacheNodes[index].baseScores == nil {
		cn.mu.Lock()
		if cn.cacheNodes[index].baseScores == nil {
			cn.cacheNodes[index].UpdateBaseScores(scorer, randomness)
		}
		cn.mu.Unlock()
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
	if idx > len(cn.actions) || idx > len(cn.baseScores) {
		log.Panicf("Invalid index for EstimatedScore: %d, actions=%d, baseScores=%d, cn.sumMCScores=%d",
			idx, len(cn.actions), len(cn.baseScores), len(cn.sumMCScores))
	}
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
	if cn.baseScores == nil {
		log.Panic("cn::Traverse(): cacheNode has no baseScores")
	}
	// Sample according to current scores.
	ii := cn.Sample()
	if depth == 0 || cn.newBoards[ii].IsFinished() {
		// If leaf node, return base score.
		return cn.baseScores[ii]
	}

	// Traverse down the sampled variation.
	if cn.baseScores == nil {
		log.Panic("cn::Traverse(): cacheNode (2) has no baseScores")
	}

	nextCN := cn.Step(ii, scorer, randomness)
	if cn.baseScores == nil {
		log.Panic("cn::Traverse(): cacheNode (3) has no baseScores")
	}

	sampledScore := -nextCN.Traverse(depth-1, scorer, priorBase, randomness)

	if cn.baseScores == nil {
		log.Panic("cn::Traverse(): cacheNode (4) has no baseScores")
	}

	// Propagate back the score.
	cn.mu.Lock()
	cn.sumMCScores[ii] += sampledScore
	cn.count[ii]++
	cn.sumExponents -= cn.exponents[ii]
	cn.exponents[ii] = math.Exp(float64(cn.EstimatedScore(ii, priorBase)) / randomness)
	cn.sumExponents += cn.exponents[ii]
	cn.mu.Unlock()

	return sampledScore
}

// Search implements the Searcher interface.
func (mcts *mctsSearcher) Search(b *Board, scorer ai.BatchScorer) (
	action Action, board *Board, score float32) {
	testCN = nil
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

		// Handle parallelism.
		maxParallel := runtime.GOMAXPROCS(0)
		if !mcts.parallelized {
			maxParallel = 1
		}
		var wg sync.WaitGroup
		semaphore := make(chan bool, maxParallel)

		// Loop over traverses.
		for time.Since(start) < mcts.maxTime {
			wg.Add(1)
			semaphore <- true
			go func() {
				cn.Traverse(mcts.maxDepth, scorer, mcts.priorBase, mcts.randomness)
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
func (mcts *mctsSearcher) ScoreMatch(
	b *Board, scorer ai.BatchScorer, actions []Action) (scores []float32) {
	var cn *cacheNode
	cn = newCacheNode(b, scorer, mcts.randomness)

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
