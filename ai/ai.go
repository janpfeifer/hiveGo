package ai

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
)

// Score for the winning side. For the loosing side it
// will be - END_GAME_SCORE.
const END_GAME_SCORE = float32(10)

// Scorer returns two scores:
//
//	value: how likely the current player is to win, in the form of
//	  a score from +10 and -10.
//	policy: Probability for each of the action. This is optional, and
//	  some models may not return it.
type Scorer interface {
	Score(board *Board, scoreActions bool) (score float32, actionProbs []float32)

	// Version returns the version of the Scorer: usually the number of features
	// used -- so system can maintain backwards compatibility.
	Version() int

	// Whether this model handles actions probabilities.
	IsActionsClassifier() bool
}

// BatchScorer is a Scorer that also handles batches.
type BatchScorer interface {
	Scorer

	// BatchScore aggregate scoring in batches -- presumable more efficient.
	BatchScore(boards []*Board, scoreActions bool) (scores []float32, actionProbsBatch [][]float32)
}

type LearnerScorer interface {
	BatchScorer

	// Learn makes the model learn from the given boards and associated boardLabels.
	// actionLabels should be given for models where IsActionsClassifier() returns true. For boards
	// that have no valid actions, that is when `len(board.Derived.Actions) == 0`, it should be nil.
	// steps is the number of times to repeat the training. If set to 0, the Learn() funciton
	// will only report loss but not actually learn.
	// If perStepCallback is given, it is called after each step, except if steps==0.
	// It returns the total loss, the loss due to the board, and the loss due to actions (if
	// also a actions classifier, otherwise 0)
	Learn(boards []*Board, boardLabels []float32, actionsLabels [][]float32,
		learningRate float32, steps int, perStepCallback func()) (loss, boardLoss, actionsLoss float32)
	Save()
	String() string
}

// Trivial implementation of a BatchScorer, wity no efficiency gains.
type BatchScorerWrapper struct {
	Scorer
}

func (s BatchScorerWrapper) BatchScore(boards []*Board, scoreActions bool) (scores []float32, actionProbsBatch [][]float32) {
	scores = make([]float32, len(boards))
	actionProbsBatch = nil
	for ii, board := range boards {
		var actionProbs []float32
		scores[ii], actionProbs = s.Score(board, scoreActions)
		if actionProbs != nil {
			if actionProbsBatch == nil {
				actionProbsBatch = make([][]float32, len(boards))
			}
			actionProbsBatch[ii] = actionProbs
		}
	}
	return
}

// Returns weather it's the end of the game, and the hard-coded score of a win/loss/draw
// for the current player if it is finished.
func EndGameScore(b *Board) (isEnd bool, score float32) {
	if !b.IsFinished() {
		return false, 0
	}
	if b.Draw() {
		return true, 0
	}
	if b.Derived.Wins[b.NextPlayer] {
		// Current player wins.
		return true, END_GAME_SCORE
	}
	// Opponent player wins.
	return true, -END_GAME_SCORE
}

// Returns a slice of float32 with one element set to 1, and all others to 0.
func OneHotEncoding(total, selected int) (vec []float32) {
	vec = make([]float32, total)
	if total > 0 {
		vec[selected] = 1
	}
	return
}
