package features

import (
	"fmt"
	"k8s.io/klog/v2"

	. "github.com/janpfeifer/hiveGo/internal/state"
)

// This file handles generation of feature per position on the board.

// PositionId is an enum of position feature ids.
type PositionId uint8

// Features Per Position:
const (
	// +1 for current player, -1 for opponent player or 0 if there are no pieces
	// at this position.
	POS_FEATURE_PLAYER_OWNER PositionId = iota

	// This is the one-hot encoding of the piece on the top of the stack.
	POS_FEATURE_PIECE_ONE_HOT = POS_FEATURE_PLAYER_OWNER + 1

	// Whether the piece is removable without breaking the hive.
	POS_FEATURE_IS_PIECE_REMOVABLE = POS_FEATURE_PIECE_ONE_HOT + int(NumPieceTypes)

	// Player owner of the stack pieces (from top to bottom), except very bottom piece.
	// Notice all will at beatles, except maybe the last one, which is represented separatedly.
	// Up to 3 (there are 4 beatles in the game, the first would be on top).
	// +1 for current player, -1 for opponent player or 0 if there are no pieces
	// at this position.
	POS_FEATURE_STACKED_BEATLE_OWNER       = POS_FEATURE_IS_PIECE_REMOVABLE + 1
	POS_FEATURE_STACK_BOTTOM_PIECE_OWNER   = POS_FEATURE_STACKED_BEATLE_OWNER + 3
	POS_FEATURE_STACK_BOTTOM_PIECE_ONE_HOT = POS_FEATURE_STACK_BOTTOM_PIECE_OWNER + 1

	// Total dimension of the feature vector per position.
	// Current value = 16
	FEATURES_PER_POSITION = POS_FEATURE_STACK_BOTTOM_PIECE_ONE_HOT + int(NumPieceTypes)
)

// PositionSpec includes the position feature name, dimension and index in the concatenation of features.
type PositionSpec struct {
	Id  PositionId
	Dim int

	// VecIndex refers to the index in the concatenated feature vector.
	VecIndex int

	// Number of features (PositionFeaturesDim) when this feature was created.
	Version int
}

var EmptyCellFeatures = make([]float32, FEATURES_PER_POSITION)

func PositionFeatures(b *Board, pos Pos) []float32 {
	stack := b.StackAt(pos)
	if !stack.HasPiece() {
		return EmptyCellFeatures
	}
	return stackFeatures(b, pos, stack)
}

func stackFeatures(b *Board, pos Pos, stack EncodedStack) (f []float32) {
	f = make([]float32, FEATURES_PER_POSITION)
	// If there is nothing there, all features related to the pieces are zeroed already.
	if stack.HasPiece() {
		stack, player, piece := stack.PopPiece()
		f[POS_FEATURE_PLAYER_OWNER] = playerToValue(b, player)
		f[POS_FEATURE_PIECE_ONE_HOT+int(piece-1)] = 1

		// Stack information:
		var stackPos int
		for stackPos = 0; stack.HasPiece() && stackPos < 4; stackPos++ {
			stack, player, piece = stack.PopPiece()
			f[POS_FEATURE_STACKED_BEATLE_OWNER+stackPos] = playerToValue(b, player)
		}
		// PieceType at very bottom is stored separately, in the next field:
		f[POS_FEATURE_STACKED_BEATLE_OWNER+stackPos-1] = 0

		// Information about piece at the very bottom of stack -- in most cases the same as the top of the stack.
		f[POS_FEATURE_STACK_BOTTOM_PIECE_OWNER] = playerToValue(b, player)
		f[POS_FEATURE_STACK_BOTTOM_PIECE_ONE_HOT+int(piece-1)] = 1

		// Is piece removable?
		if b.Derived.RemovablePositions.Has(pos) {
			f[POS_FEATURE_IS_PIECE_REMOVABLE] = 1.0
		}
	}
	return
}

// Expects owner value + one hot encoding of piece.
func pieceFeatureToStr(f []float32) string {
	if f[0] == 0 {
		return "(None)"
	}
	player := "Current"
	if f[0] == -1 {
		player = "Opponent"
	}
	for ii := ANT; ii < LastPiece; ii++ {
		if f[int(ii)] == 1 {
			return fmt.Sprintf("%s(%s)", PieceNames[ii], player)
		}
	}
	return fmt.Sprintf("NoPiece(%s)", player)
}

func PositionFeaturesToString(f []float32) string {
	if len(f) != FEATURES_PER_POSITION {
		msg := fmt.Sprintf("Invalid Position Features: wanted dimension=%d, got dimension=%d",
			FEATURES_PER_POSITION, len(f))
		klog.Error(msg)
		return msg
	}

	return fmt.Sprintf("[Top: %s, Stack(%v), Bottom: %s / Removable=%g]",
		pieceFeatureToStr(f[POS_FEATURE_PLAYER_OWNER:]),
		f[POS_FEATURE_STACKED_BEATLE_OWNER:POS_FEATURE_STACKED_BEATLE_OWNER+3],
		pieceFeatureToStr(f[POS_FEATURE_STACK_BOTTOM_PIECE_OWNER:]),
		f[POS_FEATURE_IS_PIECE_REMOVABLE])
}

func init() {
	klog.V(1).Infof("Number of features per position = %d\n", FEATURES_PER_POSITION)
}
