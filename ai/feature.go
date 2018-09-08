package ai

import (
	"log"

	. "github.com/janpfeifer/hiveGo/state"
)

// Enum of feature.
type FeatureId int

// FeatureSetter is the signarure of a feature setter. f is the slice where to store the
// results.
// fId is the id of the
type FeatureSetter func(b *Board, fId FeatureId, f []float64)

// FeatureDef includes feature name, dimension and index in the concatenation of features.
type FeatureDef struct {
	FId  FeatureId
	Name string
	Dim  int

	// VecIndex refers to the index in the concatenated feature vector.
	VecIndex int
	Setter   FeatureSetter
}

const (
	// How many pieces are offboard, per piece type.
	F_PIECES_OFFBOARD FeatureId = iota
	F_OPP_PIECES_OFFBOARD

	// How many pieces are around the queen (0 if queen hasn't been placed)
	F_NUM_SURROUNDING_QUEEN
	F_OPP_NUM_SURROUNDING_QUEEN

	// How many pieces can move.
	F_PIECES_CAN_MOVE
	F_OPP_PIECES_CAN_MOVE

	// Last entry.
	F_NUM_FEATURES
)

var (
	// Enumeration, in order, of the features extracted by FeatureVector.
	// The VecIndex attribute is properly set during the package initialization.
	// The  "Opp" prefix refers to opponent.
	AllFeatures = [F_NUM_FEATURES]FeatureDef{
		{F_PIECES_OFFBOARD, "PiecesOffboard", 5, 0, fPiecesOffBoard},
		{F_OPP_PIECES_OFFBOARD, "OppPiecesOffboard", 5, 0, fPiecesOffBoard},

		{F_NUM_SURROUNDING_QUEEN, "NumPiecesSurroundingQueen", 1, 0, fNumSurroundingQueen},
		{F_OPP_NUM_SURROUNDING_QUEEN, "OppNumPiecesSurroundingQueen", 1, 0, fNumSurroundingQueen},

		// {F_PIECES_CAN_MOVE, "PiecesCanMove", 5, 0, fPiecesCanMove},
		// {F_OPP_PIECES_CAN_MOVE, "PiecesCanMove", 5, 0, fOppPiecesCanMove},
	}

	// AllFeaturesDim is the dimension of all features concatenated, set during package
	// initialization.
	AllFeaturesDim int
)

func init() {
	// Updates the indices of AllFeatures, and sets AllFeaturesDim.
	AllFeaturesDim = 0
	for ii := range AllFeatures {
		if AllFeatures[ii].FId != FeatureId(ii) {
			log.Fatalf("ai.AllFeatures index %d for %s doesn't match constant.",
				ii, AllFeatures[ii].Name)
		}
		AllFeatures[ii].VecIndex = AllFeaturesDim
		AllFeaturesDim += AllFeatures[ii].Dim
	}
}

// FeatureVector calculates the feature vector, of length AllFeaturesDim, for the given
// board.
func FeatureVector(b *Board) (f []float64) {
	f = make([]float64, AllFeaturesDim)
	for ii := range AllFeatures {
		AllFeatures[ii].Setter(b, AllFeatures[ii].FId, f)
	}
	return
}

func fPiecesOffBoard(b *Board, fId FeatureId, f []float64) {
	idx := AllFeatures[fId].VecIndex
	player := b.NextPlayer
	if fId == F_OPP_PIECES_OFFBOARD {
		player = b.OpponentPlayer()
	}
	for _, piece := range Pieces {
		f[idx+int(piece)] = float64(b.Available(player, piece))
	}
}

func fNumSurroudingQueen(b *Board, fId FeatureId, f []float64) {
	idx := AllFeatures[fId].VecIndex
	player := b.NextPlayer
	if fId == F_OPP_NUM_SURROUNDING_QUEEN {
		player = b.OpponentPlayer()
	}
	f[idx] = float64(b.Derived.NumSurroundingQueen[player])
}
