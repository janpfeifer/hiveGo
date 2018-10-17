// Package players provides constructos of AI players from flags.
package players

import (
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ai/search"
	"github.com/janpfeifer/hiveGo/ai/tensorflow"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf

// Player is anything that is able to play the game.
type Player interface {
	// Play returns the action chosen, the next board position and the associated score predicted.
	Play(b *Board) (action Action, board *Board, score float32)
}

// SearcherScorerPlayer is a standard set up for an AI: a searcher and
// a scorer. It implements the Player interface.
type SearcherScorerPlayer struct {
	Searcher     search.Searcher
	Scorer       ai.BatchScorer
	Learner      ai.LearnerScorer
	ModelFile    string
	Parallelized bool
}

// Play implements the Player interface: it chooses an action given a Board.
func (p *SearcherScorerPlayer) Play(b *Board) (action Action, board *Board, score float32) {
	action, board, score = p.Searcher.Search(b)
	// log.Printf("Move #%d: AI playing %v, score=%.3f", b.MoveNumber, action, score)
	// log.Printf("Features:")
	// ai.PrettyPrintFeatures(ai.FeatureVector(board))
	return
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
func NewAIPlayer(config string, parallelized bool) *SearcherScorerPlayer {
	// Break config in parts.
	params := make(map[string]string)
	parts := strings.Split(config, ",")
	if len(parts) > 1 || parts[0] != "" {
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
	}

	// Scorer.
	useTF := false
	cpu := false
	modelFile := ""
	if value, ok := params["model"]; ok {
		modelFile = value
		delete(params, "model")
	}
	if _, ok := params["tf"]; ok {
		useTF = true
		delete(params, "tf")
	}
	if _, ok := params["cpu"]; ok {
		cpu = true
		delete(params, "cpu")
	}
	var model ai.LearnerScorer
	if useTF {
		model = tensorflow.New(modelFile, cpu)
	} else {
		model = ai.NewLinearScorerFromFile(modelFile)
	}

	// Configure searcher.
	var searcher search.Searcher
	var err error

	maxDepth := -1
	var maxTime time.Duration
	maxTraverses := 200
	useUCT := false

	randomness := 0.0
	if value, ok := params["max_depth"]; ok {
		delete(params, "max_depth")
		maxDepth, err = strconv.Atoi(value)
		if err != nil {
			log.Panicf("Invalid AI value '%s' for max_depth: %s", value, err)
		}
	}
	if value, ok := params["randomness"]; ok {
		delete(params, "randomness")
		randomness, err := strconv.ParseFloat(value, 64)
		if err != nil || randomness <= 0.0 {
			log.Panicf("Invalid AI value '%s' for randomness: %s", value, err)
		}
	}
	if value, ok := params["max_time"]; ok {
		delete(params, "max_time")
		secs, err := strconv.ParseFloat(value, 64)
		if err != nil {
			log.Panicf("Invalid AI value '%s' for max_time: %s", value, err)
		}
		maxTime = time.Microsecond * time.Duration(1e6*secs)
	}
	if value, ok := params["max_traverses"]; ok {
		delete(params, "max_traverses")
		maxTraverses, err = strconv.Atoi(value)
		if err != nil {
			log.Panicf("Invalid AI value '%s' for max_traverse: %s", value, err)
		}
	}
	if _, ok := params["use_uct"]; ok {
		delete(params, "use_uct")
		useUCT = true
		if parallelized {
			glog.Errorf("UCT version of MCST ('use_uct') cannot be parallelized.")
			parallelized = false // UCT doesnt' work parallelized (not yet at least)
		}
	}

	if _, ok := params["mcts"]; ok {
		delete(params, "mcts")
		if maxDepth < 0 {
			maxDepth = 8
		}
		if maxTime == 0 {
			maxTime = 5 * time.Second
		}
		if randomness == 0.0 {
			// Number found after a few experiments. TODO: can it be improved ? Or have
			// another model to decide this ?
			randomness = 0.5
		}
		searcher = search.NewMonteCarloTreeSearcher(
			maxDepth, maxTime, maxTraverses, useUCT, model, randomness, parallelized)
	}
	if _, ok := params["ab"]; ok {
		delete(params, "ab")
		// Since it is default, no need to do anything.
		searcher = nil
	}
	if searcher == nil {
		if maxDepth < 0 {
			maxDepth = 3
		}

		if randomness <= 0 {
			searcher = search.NewAlphaBetaSearcher(maxDepth, parallelized, model)
		} else {
			// Randomized searcher.
			searcher = search.NewAlphaBetaSearcher(maxDepth, false, model)
			searcher = search.NewRandomizedSearcher(searcher, model, randomness)
		}
	}

	// Check that all parameters were processed.
	if len(params) > 0 {
		for key, value := range params {
			log.Printf("Unknown parameter setting '%s=%s'", key, value)
		}
		panic("Cannot continue")
	}

	return &SearcherScorerPlayer{
		Searcher:     searcher,
		Scorer:       model,
		Learner:      model,
		ModelFile:    modelFile,
		Parallelized: parallelized,
	}
}
