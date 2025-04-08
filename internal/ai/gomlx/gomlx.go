// Package gomlx implements BoardScorer (for Alpha-Beta-Pruning and MinMax searchers) and
// PolicyScorer (for MCTS/AlphaZero) searchers.
//
// It separate the HiveGo BoardScorer and PolicyScorer implementation from the GoMLX models that support them --
// for now only FNN (Feedforward Neural Network) models are implemented.
package gomlx

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"sync"
	"weak"
)

type ModelType int

const (
	ModelNone ModelType = iota
	ModelFNN
)

//go:generate go tool enumer -type=ModelType -trimprefix=ValueModel -transform=snake -values -text -json -yaml scorer.go

var (
	// Backend is a singleton, the same for all players.
	backend = sync.OnceValue(func() backends.Backend { return backends.New() })

	// muNewClient is a Mutex used to synchronize access to GoMLX client initialization
	// or related critical sections.
	muNewClient sync.Mutex

	// Cache of models: per model type / checkpoint name.
	muModelsCache sync.Mutex
	modelsCache   = make(map[string]map[string]weak.Pointer[ai.ValueScorer])
)

const notSpecified = "#<not_specified>"

// New creates a new GoMLX based scorer/learner if the supported model is selected in parameters.
// Currently selected model names:
//
//   - "fnn": the parameter should map to the file name with the model weights. If the file doesn't
//     exist, a model is created with random weights. It's a BoardScorer.
//
// If no known model type is configured, it returns nil, nil.
func New(params parameters.Params) (ai.ValueScorer, error) {
	muModelsCache.Lock()
	defer muModelsCache.Unlock()

	for _, modelType := range ModelTypeValues() {
		if modelType == ModelNone {
			continue
		}
		key := modelType.String()
		filePath, _ := parameters.PopParamOr(params, key, notSpecified)
		if filePath == notSpecified {
			continue
		}

		// Check cache for previously created models.
		cachePerModelType, found := modelsCache[key]
		if found {
			if weakPtr, found := cachePerModelType[filePath]; found {
				if strongPtr := weakPtr.Value(); strongPtr != nil {
					return *strongPtr, nil
				}
				// weak scorer has been collected.
				delete(cachePerModelType, filePath)
			}
		} else {
			cachePerModelType = make(map[string]weak.Pointer[ai.ValueScorer])
			modelsCache[key] = cachePerModelType
		}

		// Create model and context.
		var boardScorer ai.ValueScorer
		var err error
		switch modelType {
		case ModelFNN:
			model := NewFNN()
			boardScorer, err = newBoardScorer(modelType, filePath, model, params)
			if err != nil {
				return nil, err
			}
		default:
			return nil, errors.Errorf("model type %s defined but not implemented", modelType)
		}

		// Cache resulting scorer: some awkward casting, but it works.
		cachePerModelType[filePath] = weak.Make(&boardScorer)
		klog.V(1).Infof("Created new scorer %s", boardScorer)
		return boardScorer, nil
	}
	return nil, nil
}

// init registers New as a potential scorer, so end users can use it.
func init() {
	players.RegisteredScorers = append(players.RegisteredScorers,
		func(params parameters.Params) (ai.ValueScorer, error) {
			scorer, err := New(params)
			if scorer == nil || err != nil {
				return nil, err
			}
			return scorer, nil
		})
}

// extractParams and write them as context hyperparameters
func extractParams(modelName string, params parameters.Params, ctx *context.Context) error {
	var err error
	ctx.EnumerateParams(func(scope, key string, valueAny any) {
		if err != nil {
			// If error happened skip the rest.
			return
		}
		if scope != context.RootScope {
			return
		}
		switch defaultValue := valueAny.(type) {
		case string:
			value, _ := parameters.PopParamOr(params, key, defaultValue)
			ctx.SetParam(key, value)
		case int:
			value, newErr := parameters.PopParamOr(params, key, defaultValue)
			if newErr != nil {
				err = errors.WithMessagef(newErr, "parsing %q (int) for model %s", key, modelName)
				return
			}
			ctx.SetParam(key, value)
		case float64:
			value, newErr := parameters.PopParamOr(params, key, defaultValue)
			if newErr != nil {
				err = errors.WithMessagef(newErr, "parsing %q (float64) for model %s", key, modelName)
				return
			}
			ctx.SetParam(key, value)
		case float32:
			value, newErr := parameters.PopParamOr(params, key, defaultValue)
			if newErr != nil {
				err = errors.WithMessagef(newErr, "parsing %q (float32) for model %s", key, modelName)
				return
			}
			ctx.SetParam(key, value)
		case bool:
			value, newErr := parameters.PopParamOr(params, key, defaultValue)
			if newErr != nil {
				err = errors.WithMessagef(newErr, "parsing %q (bool) for model %s", key, modelName)
				return
			}
			ctx.SetParam(key, value)
		default:
			err = errors.Errorf("model %s parameter %q is of unknown type %T", modelName, key, defaultValue)
		}
	})
	return err
}
