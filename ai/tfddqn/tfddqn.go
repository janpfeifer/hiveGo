// tfddqn Learns with Double Q-Learning.
// See paper in https://arxiv.org/abs/1509.06461
package tfddqn

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/ai/tensorflow"
	. "github.com/janpfeifer/hiveGo/state"
)

// UseType defines which use is being made of the Scorer: the
// DDQN algorithm depends on using the correct one for
// each task.
type UseType int

const (
	ForSearch UseType = 0
	ForLearn UseType = 0
	ForScore UseType = 1
)

type Scorer struct{
	models [2]*tensorflow.Scorer

	// active is the model currently selected to score and learn.
	active int

	// usage specified the current usage. The final model used depends
	// on this value.
	usage UseType
}

// New returns a new scorer, build from two tensorflow models
// that alternate during training: one used to search the best
// play, and the other to return its estimate of score.
func New(basename string) (scorer *Scorer) {
	scorer = &Scorer{}
	names := []string{
		basename + "/a",
		basename + "/b",
	}
	for ii, name := range names {
		scorer.models[ii] = tensorflow.New(name, 1, false)
	}
	return
}

// SwapModels alternate the model currently used to search/learn.
func (s *Scorer) SwapModels() { s.active = 1-s.active }

// SetUsage specifies the model usage.
func (s *Scorer) SetUsage(u UseType) { s.usage = u }

func (s *Scorer) modelIdx() int { return s.active ^ int(s.usage) }
func (s *Scorer) model() *tensorflow.Scorer { return s.models[s.modelIdx()] }


// Score implements ai.Scorer interface by calling the selected model's implementation.
func (s *Scorer) Score(board *Board, scoreActions bool) (score float32, actionProbs []float32) {
	model := s.model()
	return model.Score(board, scoreActions)
}

// Version implements ai.Scorer interface by calling the selected model's implementation.
func (s *Scorer) Version() int {
	model := s.model()
	return model.Version()
}

// IsActionsClassifier implements ai.Scorer interface by calling the selected model's implementation.
func (s *Scorer) IsActionsClassifier() bool {
	model := s.model()
	return model.IsActionsClassifier()
}

// BatchScore implements BatchScorer interface by calling the selected model's implementation.
func (s *Scorer) BatchScore(boards []*Board, scoreActions bool) (scores []float32, actionProbsBatch [][]float32) {
	model := s.model()
	return model.BatchScore(boards, scoreActions)
}

// Learn implements ai.LearnerScorer by calling the selected model's implementation.
func (s *Scorer) Learn(
	boards []*Board, boardLabels []float32, actionsLabels [][]float32,
	_ float32, epochs int, perStepCallback func()) (
	loss, boardLoss, actionsLoss float32) {
	model := s.model()
	return model.Learn(boards, boardLabels, actionsLabels, 0, epochs, perStepCallback)
}

// Save implements ai.LearnerScorer by calling the selected model's implementation.
func (s *Scorer) Save() {
	model := s.model()
	model.Save()
}

// String implements ai.LearnerScorer.
func (s *Scorer) String() string {
	var selections [2]string
	selections[s.modelIdx()] = "(*)"
	return fmt.Sprintf("TF DDQN: models in %s%s and %s%s", selections[0], s.models[0].Basename,
		selections[1], s.models[1].Basename)
}


