package gomlx

import (
	"bytes"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"slices"
	"sync"
)

type ModelType int

const (
	ModelNone ModelType = iota
	ModelFNN
)

//go:generate go tool enumer -type=ModelType -trimprefix=ValueModel -transform=snake -values -text -json -yaml scorer.go

// Scorer implements a generic GoMLX scorer for the Hive game.
// It implements ai.BoardScorer, ai.BatchBoardScorer and ai.LearnerScorer.
//
// It is just a wrapper around on of the models implemented.
type Scorer struct {
	Type ModelType

	// model used by the Scorer.
	model ValueModel

	// Executors.
	scoreExec, lossExec, trainStepExec *context.Exec

	// checkpoint handler, if model is being saved/loaded to/from disk.
	checkpoint *checkpoints.Handler

	// checkpointsToKeep is the number of copies of older checkpoints to keep around.
	// Default to 10.
	checkpointsToKeep int

	// Hyperparameters cached values: they should also be set in modelCtx.
	batchSize int

	// muLearning "write" for learning, and "read" for scoring.
	muLearning sync.RWMutex

	// optimizer used when training the model.
	// ?Should this be owned by the model itself?
	optimizer optimizers.Interface

	// muSave makes saving sequential.
	muSave sync.Mutex
}

var (
	// Assert Scorer is an ai.BoardScorer, an ai.BatchBoardScorer and an ai.LearnerScorer.
	_ ai.BoardScorer      = (*Scorer)(nil)
	_ ai.BatchBoardScorer = (*Scorer)(nil)
	_ ai.LearnerScorer    = (*Scorer)(nil)
)

var (
	// Backend is a singleton, the same for all players.
	backend = sync.OnceValue(func() backends.Backend { return backends.New() })

	// muNewClient is a Mutex used to synchronize access to GoMLX client initialization
	// or related critical sections.
	muNewClient sync.Mutex

	// Cache of models: per model type / checkpoint name.
	muModelsCache sync.Mutex
	modelsCache   = make(map[string]map[string]*Scorer)
)

const notSpecified = "#<not_specified>"

// New creates a new GoMLX based scorer/learner if the supported model is selected in parameters.
// Currently selected model names:
//
//   - "fnn": the parameter should map to the file name with the model weights. If the file doesn't
//     exist, a model is created with random weights.
//
// If no known model type is configured, it returns nil, nil.
func New(params parameters.Params) (*Scorer, error) {
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
		var s *Scorer
		cachePerModelType, found := modelsCache[key]
		if found {
			if s, found = cachePerModelType[filePath]; found {
				return s, nil
			}
		} else {
			cachePerModelType = make(map[string]*Scorer)
			modelsCache[key] = cachePerModelType
		}

		// Create model and context.
		s = &Scorer{Type: modelType}
		switch modelType {
		case ModelFNN:
			s.model = NewFNN()
		default:
			return nil, errors.Errorf("model type %s defined but not implemented", modelType)
		}

		// Help if requested.
		if slices.Index([]string{"help", "--help", "-help", "-h"}, filePath) != -1 {
			s.writeHyperparametersHelp()
			return nil, fmt.Errorf("model type %s help requested", modelType)
		}

		// Number of checkpoints to keep.
		var err error
		s.checkpointsToKeep, err = parameters.PopParamOr(params, "keep", 10)
		if err != nil {
			return nil, err
		}

		// Create checkpoint, and load it if it exists.
		if filePath != "" {
			if err = s.createCheckpoint(filePath); err != nil {
				return nil, errors.WithMessagef(err, "failed to build checkpoint for model %s in path %s",
					modelType, filePath)
			}
		}

		// Create the backend.
		_ = backend()

		// Overwrite hyperparameters from given params.
		err = s.extractParams(params)
		if err != nil {
			return nil, err
		}
		ctx := s.model.Context()
		s.batchSize = context.GetParamOr(ctx, "batch_size", 100)

		// Create optimizer to be used in training.
		s.optimizer = optimizers.FromContext(ctx)

		// Setup scoreExec executor.
		muNewClient.Lock()
		defer muNewClient.Unlock()
		s.scoreExec = context.NewExec(backend(), ctx,
			func(ctx *context.Context, inputs []*graph.Node) *graph.Node {
				// Remove last axis with dimension 1.
				ctx = ctx.Checked(false)
				return graph.Squeeze(s.model.ForwardGraph(ctx, inputs), -1)
			})
		s.lossExec = context.NewExec(backend(), ctx,
			func(ctx *context.Context, inputsAndLabels []*graph.Node) *graph.Node {
				inputs := inputsAndLabels[:len(inputsAndLabels)-1]
				labels := inputsAndLabels[len(inputsAndLabels)-1]
				if labels.Rank() == 1 {
					// Add the last axes with dimension 1.
					labels = graph.ExpandAxes(labels, -1)
				}
				loss := s.model.LossGraph(ctx, inputs, labels)
				if !loss.IsScalar() {
					// Some losses may return one value per example of the batch.
					loss = graph.ReduceAllMean(loss)
				}
				return loss
			})
		s.trainStepExec = context.NewExec(backend(), s.model.Context(),
			func(ctx *context.Context, inputsAndLabels []*graph.Node) *graph.Node {
				inputs := inputsAndLabels[:len(inputsAndLabels)-1]
				labels := inputsAndLabels[len(inputsAndLabels)-1]
				g := labels.Graph()
				ctx.SetTraining(g, true)
				loss := s.model.LossGraph(ctx, inputs, labels)
				s.optimizer.UpdateGraph(ctx, g, loss)
				train.ExecPerStepUpdateGraphFn(ctx, g)
				return loss
			})

		// Force creating/loading of variables without race conditions first.
		board := state.NewBoard()
		_ = s.BoardScore(board)

		// Cache resulting scorer.
		cachePerModelType[filePath] = s

		klog.V(1).Infof("Created new scorer %s", s)

		return s, nil
	}
	return nil, nil
}

// init registers New as a potential scorer, so end users can use it.
func init() {
	players.RegisteredScorers = append(players.RegisteredScorers,
		func(params parameters.Params) (ai.BoardScorer, error) {
			scorer, err := New(params)
			if scorer == nil || err != nil {
				return nil, err
			}
			return scorer, nil
		})
}

// String implements fmt.Stringer and ai.Scorer.
func (s *Scorer) String() string {
	if s == nil {
		return "<nil>[GoMLX]"
	}
	if s.checkpoint == nil {
		return fmt.Sprintf("%s[GoMLX]", s.Type)
	}
	return fmt.Sprintf("%s[GoMLX]@%s", s.Type, s.checkpoint.Dir())
}

// BoardScore implements ai.BoardScorer.
func (s *Scorer) BoardScore(board *state.Board) float32 {
	return s.BatchBoardScore([]*state.Board{board})[0]
}

// BatchBoardScore implements ai.BatchBoardScorer.
func (s *Scorer) BatchBoardScore(boards []*state.Board) []float32 {
	inputs := s.model.CreateInputs(boards)

	s.muLearning.RLock()
	defer s.muLearning.RUnlock()
	donatedInputs := generics.SliceMap(inputs, func(t *tensors.Tensor) any {
		return graph.DonateTensorBuffer(t, backend())
	})

	scoresT := s.scoreExec.Call(donatedInputs...)[0]
	scores := scoresT.Value().([]float32)
	// Remove any padding:
	return scores[:len(boards)]
}

// Learn implements ai.LearnerScorer, and trains model with the new boards and its labels.
// It returns the lossExec.
func (s *Scorer) Learn(boards []*state.Board, boardLabels []float32) (loss float32) {
	//fmt.Printf("Learn(%d boards)\n", len(boards))
	s.muLearning.Lock()
	defer s.muLearning.Unlock()
	lossT := s.trainStepExec.Call(s.createInputsAndLabels(boards, boardLabels)...)[0]
	return tensors.ToScalar[float32](lossT)
}

// Loss returns a measure of lossExec for the model -- whatever it is.
func (s *Scorer) Loss(boards []*state.Board, boardLabels []float32) (loss float32) {
	s.muLearning.RLock()
	defer s.muLearning.RUnlock()
	lossT := s.lossExec.Call(s.createInputsAndLabels(boards, boardLabels)...)[0]
	return tensors.ToScalar[float32](lossT)
}

func (s *Scorer) createInputsAndLabels(boards []*state.Board, boardLabels []float32) []any {
	inputs := s.model.CreateInputs(boards)
	inputs = append(inputs, s.model.CreateLabels(boardLabels))
	donatedInputs := generics.SliceMap(inputs, func(t *tensors.Tensor) any {
		return graph.DonateTensorBuffer(t, backend())
	})
	return donatedInputs
}

// Save should save the model.
func (s *Scorer) Save() error {
	if s.checkpoint == nil {
		klog.Warningf("This %s model is not associated to a checkpoint directory,  not saving", s.Type)
		return nil
	}
	return s.checkpoint.Save()
}

// BatchSize returns the recommended batch size and implements ai.LearnerScorer.
func (s *Scorer) BatchSize() int {
	return s.batchSize
}

// writeHyperparametersHelp enumerates all the hyperparameters set in the context.
func (s *Scorer) writeHyperparametersHelp() {
	buf := &bytes.Buffer{}
	_, _ = fmt.Fprintf(buf, "ValueModel %s parameters:\n", s.Type)
	s.model.Context().EnumerateParams(func(scope, key string, value any) {
		if scope != context.RootScope {
			return
		}
		_, _ = fmt.Fprintf(buf, "\t%q: default value is %v\n", key, value)
	})
	klog.Info(buf)
}

// extractParams and write them as context hyperparameters
func (s *Scorer) extractParams(params parameters.Params) error {
	ctx := s.model.Context()
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
				err = errors.WithMessagef(newErr, "parsing %q (int) for model %s", key, s.Type)
				return
			}
			ctx.SetParam(key, value)
		case float64:
			value, newErr := parameters.PopParamOr(params, key, defaultValue)
			if newErr != nil {
				err = errors.WithMessagef(newErr, "parsing %q (float64) for model %s", key, s.Type)
				return
			}
			ctx.SetParam(key, value)
		case float32:
			value, newErr := parameters.PopParamOr(params, key, defaultValue)
			if newErr != nil {
				err = errors.WithMessagef(newErr, "parsing %q (float32) for model %s", key, s.Type)
				return
			}
			ctx.SetParam(key, value)
		case bool:
			value, newErr := parameters.PopParamOr(params, key, defaultValue)
			if newErr != nil {
				err = errors.WithMessagef(newErr, "parsing %q (bool) for model %s", key, s.Type)
				return
			}
			ctx.SetParam(key, value)
		default:
			err = errors.Errorf("model %s parameter %q is of unknown type %T", s.Type, key, defaultValue)
		}
	})
	return err
}

func (s *Scorer) createCheckpoint(filePath string) error {
	var err error
	s.checkpoint, err = checkpoints.
		Build(s.model.Context()).
		Dir(filePath).
		Immediate().
		Keep(10).
		Done()
	return err
}
