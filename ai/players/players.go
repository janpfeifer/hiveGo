// Package players provides constructos of AI players from flags.
package players

import (
	"log"
	"strconv"
	"strings"

	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ai/search"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf

// Player is anything that is able to play the game.
type Player interface {
	Play(b *Board) Action
}

// SearcherScorePlayer is a standard set up for an AI: a searcher and
// a scorer. It implements the Player interface.
type SearcherScorePlayer struct {
	Searcher search.Searcher
	Scorer   ai.BatchScorer
}

// Play implements the Player interface: it chooses an action given a Board.
func (p *SearcherScorePlayer) Play(b *Board) Action {
	action, _, _ := p.Searcher.Search(b, p.Scorer)
	// log.Printf("Move #%d: AI playing %v, score=%.3f", b.MoveNumber, action, score)
	// log.Printf("Features:")
	// ai.PrettyPrintFeatures(ai.FeatureVector(board))
	return action
}

// NewAIPlayer creates a new AI player given the configuration string.
//
// Args:
//   config: comma-separated list of parameter. The following parameter are known.
//       * max_depth: Max depth for alpha-beta-prunning or MCST algorithms. Defaults to 3,
//         for ab and to 8 for MCST.
//       * ab: Selects the alpha-beta-prunning algorithm.
//       * randomness: Adds a layer of randomness in the search: the first level choice is
//         distributed according to a softmax of the scores of each move, divided by this value.
//         So lower values (closer to 0) means less randomness, higher value means more randomness,
//         hence more exploration.
//
func NewAIPlayer(config string) *SearcherScorePlayer {
	// Break config in parts.
	params := make(map[string]string)
	parts := strings.Split(config, ",")
	for _, part := range parts {
		subparts := strings.Split(part, "=")
		if len(subparts) == 1 {
			params[subparts[0]] = ""
		} else if len(subparts) == 2 {
			params[subparts[0]] = subparts[1]
		} else {
			log.Panicf("In AI configuration: cannot parse '%s'", part)
		}
	}

	// Configure searcher.
	var searcher search.Searcher
	var err error
	max_depth := -1
	if value, ok := params["max_depth"]; ok {
		delete(params, "max_depth")
		max_depth, err = strconv.Atoi(value)
		if err != nil {
			log.Panicf("Invalid AI value '%s' for ab_depth: %s", value, err)
		}
	}
	if _, ok := params["ab"]; ok {
		delete(params, "ab")
		// Since it is default, no need to do anything.
		searcher = nil
	}
	if searcher == nil {
		if max_depth < 0 {
			max_depth = 3
		}
		searcher = search.NewAlphaBetaSearcher(max_depth)
	}

	// Randomized searcher.
	if value, ok := params["randomness"]; ok {
		delete(params, "randomness")
		randomness, err := strconv.ParseFloat(value, 64)
		if err != nil || randomness <= 0.0 {
			log.Panicf("Invalid AI value '%s' for randomness: %s", value, err)
		}
		searcher = search.NewRandomizedSearcher(searcher, randomness)
	}

	if len(params) > 0 {
		for key, value := range params {
			log.Printf("Unknown parameter setting '%s=%s'", key, value)
		}
		panic("Cannot continue")
	}

	return &SearcherScorePlayer{
		Searcher: searcher,
		Scorer:   ai.ManualV0,
	}
}
