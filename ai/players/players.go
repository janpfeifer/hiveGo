// Package players provides constructos of AI players from flags.
package players

import (
	"log"

	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ai/search"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf

// Anything that is able to play the game.
type Player interface {
	Play(b *Board) Action
}

type searcherScorePlayer struct {
	searcher search.Searcher
	scorer   ai.Scorer
}

func NewAIPlayer() Player {
	return nil
}
