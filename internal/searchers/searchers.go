package searchers

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
)

var (
	// IdleChan is used for collaborative concurrency (in Wasm), and if used,
	// should be read in between every "chunk of processing".
	// This allows for processing to happen in the idle callbacks when running in javascript (via GopherJS).
	IdleChan chan bool
)

// Searcher is the interface that any of the search algorithms
// must adhere to be valid.
type Searcher interface {
	// Search returns the next action to take on the given board, along with the updated Board (after taking the action)
	// and the expected score of taking that action.
	Search(board *Board) (bestAction Action, bestBoard *Board, bestScore float32, err error)

	// String prints the name of the searcher.
	String() string

	// SetCooperative requests the searcher to make occasional calls to yieldFn.
	// This is required when running in the Browser, where there isn't real parallelism, and we don't want the
	// UI to become irresponsive.
	SetCooperative(yieldFn func())
}

// SearcherWithPolicy returns also a policy, a probability distribution over the actions.
type SearcherWithPolicy interface {
	SearchWithPolicy(board *Board) (action Action, nextBoard *Board, score float32, policy []float32, err error)
}
