package gomlx

import (
	"fmt"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"sync"
)

type ModelType int

const (
	ModelNone ModelType = iota
	ModelFNN
)

//go:generate go tool enumer -type=ModelType -trimprefix=Model -transform=snake -values -text -json -yaml model.go

// Scorer implements a generic GoMLX scorer for the Hive game.
// It implements ai.BoardScorer, ai.BatchBoardScorer and ai.LearnerScorer.
//
// It is just a wrapper around on of the models implemented.
type Scorer struct {
	Type ModelType

	// model used by the Scorer.
	model Model

	// modelCtx holds the model variables and hyperparameters.
	modelCtx *context.Context

	// Hyperparameters cached values: they should also be set in modelCtx.
	batchSize int

	// Linearize training.
	muLearning sync.Mutex

	// FileName where to save/load the model from.
	FileName string
	muSave   sync.Mutex
}

var (
	// Assert Scorer is an ai.BoardScorer, an ai.BatchBoardScorer and an ai.LearnerScorer.
	_ ai.BoardScorer      = (*Scorer)(nil)
	_ ai.BatchBoardScorer = (*Scorer)(nil)
	_ ai.LearnerScorer    = (*Scorer)(nil)
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
	for _, modelType := range ModelTypeValues() {
		if modelType == ModelNone {
			continue
		}
		key := modelType.String()
		filePath, _ := parameters.PopParamOr(params, key, notSpecified)
		if filePath == notSpecified {
			continue
		}

		// Create model and context.
		s := &Scorer{Type: modelType}
		switch modelType {
		case ModelFNN:
			s.model = &FNN{}
		default:
			return nil, errors.Errorf("model type %s defined but not implemented", modelType)
		}
		s.modelCtx = s.model.CreateContext()

		// Create checkpoint, and load it if it exists.

		// Overwrite hyperparameters from given params.

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
		return "<nil>"
	}
	return fmt.Sprintf("%s (path: %s)", s.Type, s.FileName)
}

// BoardScore implements ai.BoardScorer.
func (s *Scorer) BoardScore(board *state.Board) float32 {
	return 0.0
	//return s.ScoreFeatures(features.ForBoard(board, s.Version()))
}

// BatchBoardScore implements ai.BatchBoardScorer.
func (s *Scorer) BatchBoardScore(boards []*state.Board) (scores []float32) {
	scores = make([]float32, len(boards))
	for ii, board := range boards {
		scores[ii] = s.BoardScore(board)
	}
	return
}

// Learn implements ai.LearnerScorer, and trains model with the new boards and its labels.
// It returns the loss.
func (s *Scorer) Learn(boards []*state.Board, boardLabels []float32) (loss float32) {
	//fmt.Printf("Learn(%d boards)\n", len(boards))
	s.muLearning.Lock()
	defer s.muLearning.Unlock()
	return 0
}

// Loss returns a measure of loss for the model -- whatever it is.
func (s *Scorer) Loss(boards []*state.Board, boardLabels []float32) (loss float32) {
	return 0
}

// Save should save the model.
func (s *Scorer) Save() error {
	return nil
}

// BatchSize returns the recommended batch size and implements ai.LearnerScorer.
func (s *Scorer) BatchSize() int {
	return s.batchSize
}
