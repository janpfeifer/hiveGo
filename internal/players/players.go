// Package players provides a factory of AI players from flags.
// It also allows player providers to register themselves.
package players

import (
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/janpfeifer/hiveGo/internal/searchers"
	. "github.com/janpfeifer/hiveGo/internal/state"
)

type ModuleType int

const (
	ScorerType ModuleType = iota
	SearcherType
)

// Player is anything that is able to play the game.
type Player interface {
	// Play returns the action chosen, the next board position (after the action is taken)
	// and optionally the current board scores predicted (this can be used for interactive training).
	Play(board *Board) (action Action, nextBoard *Board, score float32, actionsScores []float32)

	// Finalize is called at the end of a match.
	Finalize()
}

// ScorerBuilder builds an ai.BoardScorer if the corresponding parameter(s) is set. E.g.: "linear" or "linear=v0" would
// create a linear scorer with the default or "v0" models respectively. It should return nil, if its parameter
// is not present.
//
// The parameters used should be removed (popped) from params.
type ScorerBuilder func(params parameters.Params) (ai.BoardScorer, error)

// SearcherBuilder builds a searchers.Searcher if the corresponding parameter is set. E.g. "ab" would create
// an alpha-beta pruning searcher. It should return nil if its parameter is not present.
//
// The parameters used should be removed (popped) from params.
type SearcherBuilder func(scorer ai.BoardScorer, params parameters.Params) (searchers.Searcher, error)

var (
	// RegisteredScorers are called in sequence to check if they apply to the params given.
	// They should either return nil if the params don't refer to them, or the ai.BoardScorer built.
	//
	// If the scorer returned also implements an ai.LearnerScorer, it can be used by the trainer
	// tool.
	RegisteredScorers []ScorerBuilder

	// RegisteredSearchers are called in sequence to check if they apply to the params given.
	// They should either return nil if the params don't refer to them, or the searchers.Searcher built.
	RegisteredSearchers []SearcherBuilder
)

var (
	// DefaultPlayerConfig is used if no configuration was given to the AI. The value may be changed by the
	// UI built.
	DefaultPlayerConfig = "linear,ab,max_depth=2"
)
