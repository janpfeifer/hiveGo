// Package _default registers the default players that can be included in any
// front-end for hiveGo.
//
// It should be imported only for its initialization, so like:
//
//	import _ "github.com/janpfeifer/hiveGo/internal/players/default"
//
// Currently, it includes a linear model and GoMLX for Go (but not XLA).
// It includes both searchers: Alpha-Beta pruning and Monte Carlo Tree Search (MCTS).
package _default

import (
	"github.com/janpfeifer/hiveGo/internal/ai/linear"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/searchers/alphabeta"
	"github.com/janpfeifer/hiveGo/internal/searchers/mcts"

	_ "github.com/gomlx/gomlx/backends/simplego"
	_ "github.com/janpfeifer/hiveGo/internal/ai/gomlx"
)

func init() {
	// Register default scorers and searchers.
	players.RegisteredScorers = append(players.RegisteredScorers, linear.NewFromParams)
	players.RegisteredSearchers = append(players.RegisteredSearchers,
		alphabeta.NewFromParams, mcts.NewFromParams)
}
