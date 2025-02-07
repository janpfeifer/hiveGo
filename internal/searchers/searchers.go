package searchers

import (
	"github.com/janpfeifer/hiveGo/internal/ai"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"math"
	"math/rand"
	"slices"
	"sort"
)

var (
	// IdleChan if created is read before each chunk of search is
	// done. This allows for processing to happens in the idle callbacks
	// when running in javascript (via GopherJS)
	IdleChan chan bool
)

// Searcher is the interface that any of the search algorithms
// must adhere to be valid.
type Searcher interface {
	// Search returns the next action to take on the given board,
	// along with the updated Board (after taking the action) and
	// the expected score of taking that action.
	//
	// Optionally, it can also generate the score for each of the actions available to board.
	Search(board *Board) (action Action, nextBoard *Board, score float32, actionsLabels []float32)

	// ScoreMatch will score the board at each board position, starting from the current one,
	// and following each one of the actions. In the end, len(scores) == len(actions)+1.
	// It also outputs actionsLabels, which may be a one-hot-encoding, or a probability
	// distribution over the actions.
	ScoreMatch(board *Board, actions []Action) (scores []float32, actionsLabels [][]float32)
}

// ExecuteAndScoreActions enumerates each of the available actions, along with the boards
// where actions were taken and with the score for current b.NextPlayer -- not the
// next action's NextPlayer. It wil return early if any of the actions lead to
// b.NextPlayer winning.
//
// Actions returned is a deep copy and can be changed.
func ExecuteAndScoreActions(board *Board, scorer ai.BatchBoardScorer) (
	actions []Action, newBoards []*Board, scores []float32) {
	if len(board.Derived.Actions) == 0 {
		actions = []Action{{Piece: NoPiece}}
	} else {
		// Make deep copy of the actions
		actions = slices.Clone(board.Derived.Actions))
	}
	scores = make([]float32, len(actions))
	newBoards = make([]*Board, len(actions))

	// Pre-score actions that lead to end-game.
	boardsToScore := make([]*Board, 0, len(actions))
	hasWinning := 0
	for ii, action := range actions {
		newBoards[ii] = board.Act(action)
		if isEnd, score := ai.EndGameScore(newBoards[ii]); isEnd {
			// End game is treated differently.
			score = -score // Score for board.NextPlayer, not newBoards[ii].NextPlayer
			if score > 0.0 {
				hasWinning++
			}
			scores[ii] = score
		} else {
			boardsToScore = append(boardsToScore, newBoards[ii])
		}
	}

	// Player wins, no need to score the other plays.
	if hasWinning > 0 {
		// Actually we could just trim return only the winning action(s). But
		// for ML training, it's useful to return all and let it learn from it.
		// But in any cases there is no need to rescore the other
		// actions.
		return

		// To trim the non-winning actions:
		//revisedActions := make([]Action, 0, hasWinning)
		//revisedBoards := make([]*Board, 0, hasWinning)
		//revisedScores := make([]float32, 0, hasWinning)
		//for ii, action := range actions {
		//	if newBoards[ii].IsFinished() && scores[ii] > 0 {
		//		revisedActions = append(revisedActions, action)
		//		revisedBoards = append(revisedBoards, newBoards[ii])
		//		revisedScores = append(revisedScores, scores[ii])
		//	}
		//}
		//return revisedActions, revisedBoards, revisedScores
	}

	if len(boardsToScore) > 0 {
		// Score non-game ending boards.
		// TODO: Use "Principal Variation" to estimate the score.
		scored, _ := scorer.BatchBoardScore(boardsToScore, false)
		scoredIdx := 0
		for ii := range scores {
			if !newBoards[ii].IsFinished() {
				// Score for board.NextPlayer, not newBoards[ii].NextPlayer, hence
				// we take the inverse here.
				scores[ii] = -scored[scoredIdx]
				scoredIdx++
			}
		}
	}

	return
}

func SortActionsBoardsScores(actions []Action, boards []*Board, scores []float32) {
	s := &ScoresToSort{actions, boards, scores}
	sort.Sort(s)
}

// ScoresToSort provides a way to jointly sort actions, boards and scores by their scores -- larger scores come first.
type ScoresToSort struct {
	actions []Action
	boards  []*Board
	scores  []float32
}

func (s *ScoresToSort) Swap(i, j int) {
	s.actions[i], s.actions[j] = s.actions[j], s.actions[i]
	s.boards[i], s.boards[j] = s.boards[j], s.boards[i]
	s.scores[i], s.scores[j] = s.scores[j], s.scores[i]
}
func (s *ScoresToSort) Len() int           { return len(s.scores) }
func (s *ScoresToSort) Less(i, j int) bool { return s.scores[i] > s.scores[j] }

