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

type SearcherScorePlayer struct {
	Searcher search.Searcher
	Scorer   ai.BatchScorer
}

func (p *SearcherScorePlayer) Play(b *Board) Action {
	action, board, score := p.Searcher.Search(b, p.Scorer)
	log.Printf("Move #%d: AI playing %v, score=%.3f", b.MoveNumber, action, score)
	log.Printf("Features:")
	ai.PrettyPrintFeatures(ai.FeatureVector(board))
	return action
}

func NewAIPlayer() *SearcherScorePlayer {
	alphaBeta := search.NewAlphaBetaSearcher(3)
	return &SearcherScorePlayer{
		Searcher: alphaBeta, // search.NewRandomizedSearcher(alphaBeta, 1.0),
		Scorer:   ai.ManualV0,
	}
	return nil
}
