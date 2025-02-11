// Package _default registers the default players that can be included in any
// front-end for hiveGo.
//
// Currently, it includes a linear model + alpha-beta pruning.
package _default

import (
	"github.com/janpfeifer/hiveGo/internal/ai/linear"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/state"
)

func init() {
	players.RegisterModule("linear", &Linear{})
}

// Linear implements a
type Linear struct{}

// Assert Linear implements Module.
var _ players.Module = (*Linear)(nil)

// NewPlayer implements players.Module.
func (l *Linear) NewPlayer(matchId uint64, matchName string, playerNum state.PlayerNum, params map[string]string) (players.Player, error) {
	return players.NewPlayerFromScorer(linear.PreTrainedBest, matchId, matchName, playerNum, params)
}
