package searchers

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
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
	// Search returns the next action to take on the given board, along with the updated Board (after taking the action)
	// and the expected score of taking that action.
	//
	// Optionally, it can also return the score for each of the actions available on the board.
	// Some algorithms (e.g.: alpha-beta pruning) don't provide good approximations to those, so they return it nil.
	Search(board *Board) (action Action, nextBoard *Board, score float32, actionsScores []float32)

	// ScoreMatch will score the board at each board position, starting from the current one,
	// and following each one of the actions. In the end, len(scores) == len(actions)+1.
	//
	// actionsLabels is a probability distribution over the actions (or a one-hot encoding if there is
	// a winning action).
	//ScoreMatch(board *Board, actions []Action) (scores []float32, actionsLabels [][]float32)
}
