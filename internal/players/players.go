// Package players provides a factory of AI players from flags.
// It also allows player providers to register themselves.
package players

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
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
	// matchId is unique, but the Module.NewPlayer may be called twice for the same matchId, for different players,
	// if self-playing during training.
	NewPlayer(matchId uint64, playerNum PlayerNum, params map[string]string) (Player, error)
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
func New(matchId uint64, playerNum PlayerNum, config string) (Player, error) {
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

	params := splitConfigString(config)
	player, err := module.NewPlayer(matchId, playerNum, params)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create AI player %q", moduleName)
	}
	return player, nil
}

// splitConfig string to a map of keys to values, all strings.
// See GetParamOr and PopParamOr to parse values from this map.
func splitConfigString(config string) map[string]string {
	params := make(map[string]string)
	parts := strings.Split(config, ",")
	for _, part := range parts {
		subParts := strings.SplitN(part, "=", 2) // Split into up to 2 parts to handle '=' in values
		if len(subParts) == 1 {
			params[subParts[0]] = ""
		} else if len(subParts) == 2 {
			params[subParts[0]] = subParts[1]
		}
	}
	return params
}

// GetParamOr attempts to parse a parameter to the given type if the key is present, or returns the defaultValue
// if not.
//
// For bool types, a key without a value is interpreted as true.
func GetParamOr[T interface{ bool | int | float32 | float64 }](params map[string]string, key string, defaultValue T) (T, error) {
	vAny := (any)(defaultValue)
	var t T
	toT := func(v any) T { return v.(T) }
	switch vAny.(type) {
	case int:
		if value, exists := params[key]; exists && value != "" {
			parsedValue, err := strconv.Atoi(value)
			if err != nil {
				return t, errors.Wrapf(err, "failed to parse configuration %s=%q to int", key, value)
			}
			return toT(parsedValue), nil
		}
	case float32:
		if value, exists := params[key]; exists && value != "" {
			parsedValue, err := strconv.ParseFloat(value, 32)
			if err != nil {
				return t, errors.Wrapf(err, "failed to parse configuration %s=%q to float", key, value)
			}
			return toT(float32(parsedValue)), nil
		}
	case float64:
		if value, exists := params[key]; exists && value != "" {
			parsedValue, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return t, errors.Wrapf(err, "failed to parse configuration %s=%q to float", key, value)
			}
			return toT(parsedValue), nil
		}
	case bool:
		if value, exists := params[key]; exists {
			if value == "" || strings.ToLower(value) == "true" || value == "1" { // Empty value is considered "true"
				return toT(true), nil
			}
			if strings.ToLower(value) == "false" || value == "0" {
				return toT(false), nil
			}
			return defaultValue, errors.New("failed to parse bool")
		}
	}
	return defaultValue, nil
}

// PopParamOr is like GetParamOr but it also deletes from the params map the retrieved parameter.
func PopParamOr[T interface{ bool | int | float32 | float64 }](params map[string]string, key string, defaultValue T) (T, error) {
	value, err := GetParamOr(params, key, defaultValue)
	if err != nil {
		return value, err
	}
	delete(params, key)
	return value, nil
}
