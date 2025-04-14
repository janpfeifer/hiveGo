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

	// IdPositionStackedBeetlesOwner represent the owners of the stacked pieces (from top to bottom),
	// except the very bottom piece. They must all be beetles, hence the name.
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

// PositionSpec includes the position feature name, dimension and index in the concatenation of features.
type PositionSpec struct {
	// Id of the position feature.
	Id PositionId

	// Dim is the number of values used to represent this feature.
	Dim int

	// Index refers to the starting index in the concatenated position features vector.
	Index int

	// Number of features (IdPositionLast) when this feature was created.
	Version int
}

var (
	// PositionSpecs lists the dimension of each positional feature, and later it is initialized
	// the relative position in the positional features vector.
	PositionSpecs = []PositionSpec{
		{Id: IdPositionPlayerOwner, Dim: 1},
		{Id: IdPositionPieceOneHot, Dim: int(NumPieceTypes)},
		{Id: IsPositionIsPieceRemovable, Dim: 1},
		{Id: IdPositionStackedBeetlesOwner, Dim: 4},
		{Id: IdPositionStackBottomPieceOwner, Dim: 1},
		{Id: IdPositionStackBottomPiece, Dim: int(NumPieceTypes)},
	}

	// PositionFeaturesDim is the sum of the dimensions of all position features = sum(PositionSpecs.Dim).
	// It is set during initialization.
	PositionFeaturesDim int
)

func init() {
	featureIdx := 0
	for specIdx := range PositionSpecs {
		PositionSpecs[specIdx].Index = featureIdx
		featureIdx += PositionSpecs[specIdx].Dim
	}
	PositionFeaturesDim = featureIdx
}

func idxForPositionId(id PositionId) int {
	return PositionSpecs[id].Index
}

func featuresForPositionId(id PositionId, features []float32) []float32 {
	idx := idxForPositionId(id)
	return features[idx : idx+PositionSpecs[id].Dim]
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
	f = make([]float32, PositionFeaturesDim)
	idx := func(id PositionId) int { return idxForPositionId(id) }
	// If there is nothing there, all features related to the pieces are zeroed already.
	if stack.HasPiece() {
		stack, player, piece := stack.PopPiece()
		f[idx(IdPositionPlayerOwner)] = playerToValue(b, player)
		f[idx(IdPositionPieceOneHot)+int(piece-1)] = 1

		// Stack information:
		var stackPos int
		for stackPos = 0; stackPos < 4; stackPos++ {
			if stack.HasPiece() {
				stack, player, piece = stack.PopPiece()
				f[idx(IdPositionStackedBeetlesOwner)+stackPos] = playerToValue(b, player)
			} else {
				f[idx(IdPositionStackedBeetlesOwner)+stackPos] = 0
			}
		}

		// Information about piece at the very bottom of stack -- in most cases the same as the top of the stack.
		f[idx(IdPositionStackBottomPieceOwner)] = playerToValue(b, player)
		f[idx(IdPositionStackBottomPiece)+int(piece-1)] = 1

		// Is piece removable?
		if b.Derived.RemovablePositions.Has(pos) {
			f[idx(IsPositionIsPieceRemovable)] = 1.0
		}
	}
	return
}

// Expects owner value + one hot encoding of piece.
func pieceFeatureToStr(owner float32, piece []float32) string {
	if owner == 0 {
		return "(None)"
	}
	player := "Current"
	if owner == -1 {
		player = "Opponent"
	}
	for ii := ANT; ii < LastPiece; ii++ {
		if piece[int(ii-ANT)] == 1 {
			return fmt.Sprintf("%s(%s)", PieceNames[ii], player)
		}
	}
	return fmt.Sprintf("NoPiece(%s)", player)
}

func PositionFeaturesToString(rawFeatures []float32) string {
	if len(rawFeatures) != PositionFeaturesDim {
		msg := fmt.Sprintf("Invalid Position Features: wanted dimension=%d, got dimension=%d",
			PositionFeaturesDim, len(rawFeatures))
		klog.Error(msg)
		return msg
	}

	extract := func(id PositionId) []float32 {
		return featuresForPositionId(id, rawFeatures)
	}
	return fmt.Sprintf("[Top: %s / Bettles(%v), Bottom: %s / Removable=%g]",
		pieceFeatureToStr(
			extract(IdPositionPlayerOwner)[0],
			extract(IdPositionPieceOneHot)),
		extract(IdPositionStackedBeetlesOwner),
		pieceFeatureToStr(
			extract(IdPositionStackBottomPieceOwner)[0],
			extract(IdPositionStackBottomPiece)),
		extract(IsPositionIsPieceRemovable))
}

func init() {
	klog.V(1).Infof("Number of features per position = %d\n", IdPositionLast)
}
