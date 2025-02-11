// Package players provides a factory of AI players from flags.
// It also allows player providers to register themselves.
package players

import (
	"github.com/janpfeifer/hiveGo/internal/parameters"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
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
	Play(board *Board) (action Action, nextBoard *Board, score float32, actionsScores []float32)

	// Finalize is called at the end of a match.
	Finalize()
}

// Module must implement NewPlayer called at the start of a match.
// matchId is unique among matches, but the Module.NewPlayer may be called twice for the same matchId, for different players,
// if self-playing during training.
// matchName is used for logging and debugging.
type Module interface {
	NewPlayer(matchId uint64, matchName string, playerNum PlayerNum, params map[string]string) (Player, error)
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

// RegisterModule so it can be used by any of the front-ends to play HiveGo.
func RegisterModule(name string, module Module) {
	keywordToModules[name] = moduleRegistration{Name: name, Module: module}
}

var (
	// DefaultPlayerConfig is used if no configuration was given to the AI. The value may be changed by the
	// UI built.
	DefaultPlayerConfig = "linear:ab,max_depth=2"
)

// New creates a new AI player given the configuration string.
//
// Args:
//
//	config: the AI name followed by a colon (":"), followed by a comma-separated list of optional parameters with optional values associated.
//		If empty, the default is given by DefaultPlayerConfig (usually "linear:ab,max_depth=2", if not changed by the program).
//
// More details on the config are dependent on the module used.
func New(matchId uint64, matchName string, playerNum PlayerNum, config string) (Player, error) {
	if config == "" {
		config = DefaultPlayerConfig
	}

	// Find moduleName.
	moduleName := config
	if moduleSplit := strings.Index(config, ":"); moduleSplit != -1 {
		moduleName = config[:moduleSplit]
		config = config[moduleSplit+1:]
	}
	module, ok := keywordToModules[moduleName]
	if !ok {
		return nil, errors.Errorf("unknown AI player %q", moduleName)
	}

	params := parameters.NewFromConfigString(config)
	player, err := module.NewPlayer(matchId, matchName, playerNum, params)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create AI player %q", moduleName)
	}
	return player, nil
}
