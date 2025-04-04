// Package ai (Artificial Intelligence) defines standard interfaces that AIs for the game
// have to implement.
package ai

import (
	"github.com/chewxy/math32"
	. "github.com/janpfeifer/hiveGo/internal/state"
)

// WinGameScore for the winning side. For the loosing side it is -WinGameScore.
// We make these +1 and -1, so it's easy to put a tanh(x) on the output of the model to get a
// value from +1 to -1.
const WinGameScore = float32(1)

// SquashScore converts any score to a value between +WinGameScore and -WinGameScore
// by using then tanh(x) function -- a type of S curve.
func SquashScore(x float32) float32 {
	return math32.Tanh(x) * WinGameScore
}

// BoardScorer returns two scores:
//
//	value: how likely the current player is to win, in the form of
//	  a score from +10 and -10.
//	policy: Probability for each of the action. This is optional, and
//	  some models may not return it.
type BoardScorer interface {
	BoardScore(board *Board) float32
	String() string
}

// BatchBoardScorer is a BoardScorer that handles batches.
type BatchBoardScorer interface {
	BoardScorer

	// BatchBoardScore aggregate board scoring in batches, presumable more efficient.
	BatchBoardScore(boards []*Board) []float32
}

// PolicyScorer represents an AI capable of scoring both the board and individual actions.
// Notice that while the board score and the policy scores share the model -- and they are learned
// jointly, they are evaluated separately, since for the leaf states visited (most of them),
// the policy values are not needed.
type PolicyScorer interface {
	BoardScorer

	// PolicyScore returns a score (probability) for each of the action of the board.
	// In the article/paper, this is $P[s] = { p(s, a), \forall a \in s }$, where s is the state (board)
	// and $a$ is an action valid in the state $s$.
	PolicyScore(board *Board) []float32
}

// LearnerScorer is the interface used to train a BoardScorer model.
type LearnerScorer interface {
	BatchBoardScorer

	// Learn makes the model learn from the given boards and associated boardLabels.
	// It returns the training loss.
	Learn(boards []*Board, boardLabels []float32) (loss float32)

	// Loss returns a measure of loss for the model -- whatever it is.
	Loss(boards []*Board, boardLabels []float32) (loss float32)

	// Save the model being learned -- or create a new checkpoint.
	Save() error

	// BatchSize returns the batch size used by the learner.
	// It is used only as an optimization hint for the trainer.
	// If Learn is called with more examples than this, it will be split, and if smaller
	// it will be padded (or something equivalent).
	BatchSize() int

	// String returns the model/learner name.
	String() string
}

// IsEndGameAndScore returns weather it's the end of the game, and the hard-coded score of a win/loss/draw
// for the current player if it is finished.
// If isEnd is false, the score should be ignored.
func IsEndGameAndScore(b *Board) (isEnd bool, score float32) {
	if !b.IsFinished() {
		return false, 0
	}
	if b.Draw() {
		return true, 0
	}
	if b.Derived.Wins[b.NextPlayer] {
		// Current player wins.
		return true, WinGameScore
	}
	// Opponent player wins.
	return true, -WinGameScore
}

// OneHotEncoding returns a slice of float32 with one element set to 1, and all others to 0.
func OneHotEncoding(total, selected int) (vec []float32) {
	vec = make([]float32, total)
	if total > 0 {
		vec[selected] = 1
	}
	return
}
