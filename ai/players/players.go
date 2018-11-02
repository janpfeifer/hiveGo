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

// External model registration functions.
type PlayerModuleInitFn func() (data interface{})
type PlayerParameterFn func(data interface{}, key, value string)
type PlayerModuleFinalizeFn func(data interface{}, player *SearcherScorerPlayer)

// Registration of an external module for a keyword.
type externalModuleRegistration struct {
	module string
	fn     PlayerParameterFn
}

var (
	// Registered external modules.
	externalModulesInitFns     = make(map[string]PlayerModuleInitFn)
	externalModulesFinalizeFns = make(map[string]PlayerModuleFinalizeFn)
	keywordToModule            = make(map[string]externalModuleRegistration)
)

// Register function to process given parameter for given module. This allows
// external modules to change the behavior of NewAIPlayer.
// For each module, initFn will be called at the start of the parsing.
// Then paramFn is called or each key/value pair (value may be empty).
// Finally finalFn is called, where the external module can change the resulting
// player object.
func RegisterPlayerParameter(module, key string, initFn PlayerModuleInitFn, paramFn PlayerParameterFn,
	finalFn PlayerModuleFinalizeFn) {
	externalModulesInitFns[module] = initFn
	externalModulesFinalizeFns[module] = finalFn
	keywordToModule[key] = externalModuleRegistration{module, paramFn}
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
	// Initialize external modules data.
	moduleToData := make(map[string]interface{})
	for module, initFn := range externalModulesInitFns {
		moduleToData[module] = initFn()
	}

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

	// External modules parsing.
	paramsLeft := make(map[string]string)
	for key, value := range params {
		if registration, ok := keywordToModule[key]; ok {
			data := moduleToData[registration.module]
			registration.fn(data, key, value)
		} else {
			paramsLeft[key] = value
		}
	}
	params = paramsLeft

	// Shared parameters.
	player := &SearcherScorerPlayer{Parallelized: parallelized}
	if value, ok := params["model"]; ok {
		player.ModelFile = value
		delete(params, "model")
	}

	// External modules make their modifications to the player object.
	for module, finalFn := range externalModulesFinalizeFns {
		data := moduleToData[module]
		finalFn(data, player)
	}

	// Default scorer.
	if player.Scorer == nil {
		player.Learner = ai.NewLinearScorerFromFile(player.ModelFile)
		player.Scorer = player.Learner
	}

	// Configure searcher.
	var searcher search.Searcher
	var err error

	maxDepth := -1
	var maxTime time.Duration
	maxTraverses := 200
	useUCT := false
	maxScore := float32(10.0)

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
		randomness, err = strconv.ParseFloat(value, 64)
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
		if player.Parallelized {
			glog.Errorf("UCT version of MCST ('use_uct') cannot be parallelized.")
			player.Parallelized = false // UCT doesn't work parallelized (not yet at least)
		}
	}
	if value, ok := params["max_score"]; ok {
		delete(params, "max_score")
		v64, err := strconv.ParseFloat(value, 64)
		if err != nil || v64 <= 0.0 {
			log.Panicf("Invalid max_score value '%s': %s", value, err)
		}
		maxScore = float32(v64)
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
			maxDepth, maxTime, maxTraverses, useUCT, maxScore,
			player.Scorer, randomness, player.Parallelized)
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
			searcher = search.NewAlphaBetaSearcher(maxDepth, player.Parallelized, player.Scorer)
		} else {
			// Randomized searcher.
			searcher = search.NewAlphaBetaSearcher(maxDepth, false, player.Scorer)
			searcher = search.NewRandomizedSearcher(searcher, player.Scorer, randomness)
		}
	}

	// Check that all parameters were processed.
	if len(params) > 0 {
		for key, value := range params {
			log.Printf("Unknown parameter setting '%s=%s'", key, value)
		}
		panic("Cannot continue")
	}
	player.Searcher = searcher

	return player
}
