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
	// IdPositionPlayerOwner is +1 if current top position is owned by the current player,
	// -1 for opponent player or 0 if there are no pieces at this position.
	IdPositionPlayerOwner PositionId = iota

	// IdPositionPieceOneHot is the one-hot encoding of the piece on the top of the stack.
	IdPositionPieceOneHot

	// IsPositionIsPieceRemovable represents whether the piece is removable without breaking the hive.
	IsPositionIsPieceRemovable

	// IdPositionStackedBeetlesOwner represent the owners of the stacked pieces (from top to bottom), except
	// very bottom piece. They must all be beetles, hence the name.
	// Up to 3 (there are 4 beatles in the game, the first would be on top).
	// +1 for current player, -1 for opponent player or 0 if there are no pieces
	// at this position.
	IdPositionStackedBeetlesOwner

	// IdPositionStackBottomPieceOwner represents the owner of the
	IdPositionStackBottomPieceOwner
	IdPositionStackBottomPiece

	// IdPositionLast represents the next free PositionId.
	// This must remain the last entry.
	// Current value = 6
	IdPositionLast
)

var (
	// PositionSpecs lists the dimension of each positional feature, and later it is initialized
	// the relative position in the positional features vector.
	PositionSpecs = []PositionSpec{
		{Id: IdPositionPlayerOwner, Dim: 1},
		{Id: IdPositionPieceOneHot, Dim: int(NumPieceTypes)},
		{Id: IsPositionIsPieceRemovable, Dim: 1},
		{Id: IdPositionStackedBeetlesOwner, Dim: 3},
		{Id: IdPositionStackBottomPieceOwner, Dim: 1},
		{Id: IdPositionStackBottomPiece, Dim: int(NumPieceTypes)},
	}
)

func init() {
	featureIdx := 0
	for specIdx := range PositionSpecs {
		PositionSpecs[specIdx].VecIndex = featureIdx
		featureIdx += PositionSpecs[specIdx].Dim
	}

}

// PositionSpec includes the position feature name, dimension and index in the concatenation of features.
type PositionSpec struct {
	// Id of the position feature.
	Id PositionId

	// Dim is the number of values used to represent this feature.
	Dim int

	// VecIndex refers to the starting index in the concatenated feature vector.
	VecIndex int

	// Number of features (IdPositionLast) when this feature was created.
	Version int
}

var EmptyCellFeatures = make([]float32, IdPositionLast)

func PositionFeatures(b *Board, pos Pos) []float32 {
	stack := b.StackAt(pos)
	if !stack.HasPiece() {
		return EmptyCellFeatures
	}
	return stackFeatures(b, pos, stack)
}

func stackFeatures(b *Board, pos Pos, stack EncodedStack) (f []float32) {
	f = make([]float32, IdPositionLast)
	// If there is nothing there, all features related to the pieces are zeroed already.
	if stack.HasPiece() {
		stack, player, piece := stack.PopPiece()
		f[IdPositionPlayerOwner] = playerToValue(b, player)
		f[IdPositionPieceOneHot+int(piece-1)] = 1

		// Stack information:
		var stackPos int
		for stackPos = 0; stack.HasPiece() && stackPos < 4; stackPos++ {
			stack, player, piece = stack.PopPiece()
			f[IdPositionStackedBeetlesOwner+stackPos] = playerToValue(b, player)
		}
		// PieceType at very bottom is stored separately, in the next field:
		f[IdPositionStackedBeetlesOwner+stackPos-1] = 0

		// Information about piece at the very bottom of stack -- in most cases the same as the top of the stack.
		f[IdPositionStackBottomPieceOwner] = playerToValue(b, player)
		f[IdPositionStackBottomPiece+int(piece-1)] = 1

		// Is piece removable?
		if b.Derived.RemovablePositions.Has(pos) {
			f[IsPositionIsPieceRemovable] = 1.0
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
	if len(f) != IdPositionLast {
		msg := fmt.Sprintf("Invalid Position Features: wanted dimension=%d, got dimension=%d",
			IdPositionLast, len(f))
		klog.Error(msg)
		return msg
	}

	return fmt.Sprintf("[Top: %s, Stack(%v), Bottom: %s / Removable=%g]",
		pieceFeatureToStr(f[IdPositionPlayerOwner:]),
		f[IdPositionStackedBeetlesOwner:IdPositionStackedBeetlesOwner+3],
		pieceFeatureToStr(f[IdPositionStackBottomPieceOwner:]),
		f[IsPositionIsPieceRemovable])
}

func init() {
	klog.V(1).Infof("Number of features per position = %d\n", IdPositionLast)
}
