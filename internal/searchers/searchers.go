package searchers

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
)

var (
	// IdleChan if created is read before each chunk of search is done.
	// This allows for processing to happen in the idle callbacks when running in javascript (via GopherJS).
	IdleChan chan bool
)

// Searcher is the interface that any of the search algorithms
// must adhere to be valid.
type Searcher interface {
	// Search returns the next action to take on the given board, along with the updated Board (after taking the action)
	// and the expected score of taking that action.
	Search(board *Board) (bestAction Action, bestBoard *Board, bestScore float32, err error)
}

// SearcherWithPolicy returns also a policy, a probability distribution over the actions.
type SearcherWithPolicy interface {
	SearchWithPolicy(board *Board) (action Action, nextBoard *Board, score float32, policy []float32, err error)
}
