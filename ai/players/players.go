// Package players provides a factory of AI players from flags.
package players

import (
	"github.com/janpfeifer/hiveGo/ai/features"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"log"
	"strconv"
	"strings"
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
	Play(board *Board, matchName string) (action Action, nextBoard *Board, score float32, actionsScores []float32)

	// Finalize is called at the end of a match.
	Finalize()
}

// Module implements a player constructor.
type Module interface {
	// NewPlayer is called once per match.
	// If self playing, this may be called twice for the same match, once per player.
	NewPlayer(playerNum int, params map[string]string) (Player, error)
}

// moduleRegistration is a reference to the module and its name.
type moduleRegistration struct {
	Module
	Name string
}

var (
	// Registered external modules.
	keywordToModules = make(map[string]moduleRegistration)
)

// RegisterPlayerModule so it can be used by any of the front-ends to play HiveGo.
func RegisterPlayerModule(name string, module Module) {
	externalModulesInitFns[module] = initFn
	if mType == ScorerType {
		externalModulesScorerFinalizeFns[module] = finalFn
	} else {
		externalModulesSearcherFinalizeFns[module] = finalFn
	}
	keywordToModules[key] = append(keywordToModules[key],
		externalModuleRegistration{module, paramFn})
}

// MustFloat32 converts string to float32 or panic.
func MustFloat32(s, paramName string) float32 {
	v64, err := strconv.ParseFloat(s, 64)
	if err != nil {
		log.Panicf("Invalid %s value '%s': %s", paramName, s, err)
	}
	return float32(v64)
}

func MustInt(s, paramName string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		log.Panicf("Invalid value for parameter %s=%s: %v", paramName, s, err)
	}
	return v
}

// NewAIPlayer creates a new AI player given the configuration string.
//
// Args:
//
//	config: comma-separated list of parameter. The following parameter are known.
//	    * max_depth: Max depth for alpha-beta-prunning or MCST algorithms. Defaults to 3,
//	      for ab and to 8 for MCST.
//	    * ab: Selects the alpha-beta-prunning algorithm.
//	    * randomness: Adds a layer of randomness in the search: the first level choice is
//	      distributed according to a softmax of the scores of each move, divided by this value.
//	      So lower values (closer to 0) means less randomness, higher value means more randomness,
//	      hence more exploration.
func NewAIPlayer(config string, parallelized bool) *SearcherScorer {
	if config == "" {
		// Default AI.
		config = "ab,max_depth=2"
	}
	// Initialize external modules data.
	moduleToData := make(map[string]any)
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
		if registrations, ok := keywordToModules[key]; ok {
			for _, registration := range registrations {
				data := moduleToData[registration.module]
				registration.fn(data, key, value)
			}
		} else {
			paramsLeft[key] = value
		}
	}
	// Check that all parameters were processed.
	if len(paramsLeft) > 0 {
		for key, value := range paramsLeft {
			log.Printf("Unknown parameter setting '%s=%s'", key, value)
		}
		panic("Cannot continue")
	}

	// Shared parameters.
	player := &SearcherScorer{Parallelized: parallelized}

	// External modules make their modifications to the player object.
	for module, finalFn := range externalModulesScorerFinalizeFns {
		data := moduleToData[module]
		finalFn(data, player)
	}

	// Default scorer.
	if player.Scorer == nil {
		player.Learner = features.NewLinearScorerFromFile(player.ModelPath)
		player.Scorer = player.Learner
	}

	// External modules make their modifications to the player object.
	for module, finalFn := range externalModulesSearcherFinalizeFns {
		data := moduleToData[module]
		finalFn(data, player)
	}

	if player.Searcher == nil {
		log.Panicf("Searcher not specified.")
	}

	return player
}
