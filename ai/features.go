package ai

import (
	"fmt"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"log"
)

// Enum of feature.
type FeatureId int

// FeatureSetter is the signature of a feature setter. f is the slice where to store the
// results.
// fId is the id of the
type FeatureSetter func(b *Board, def *FeatureDef, f []float32)

const (
	// How many pieces are offboard, per piece type.
	F_NUM_OFFBOARD FeatureId = iota
	F_OPP_NUM_OFFBOARD

	// How many pieces are around the queen (0 if queen hasn't been placed)
	F_NUM_SURROUNDING_QUEEN
	F_OPP_NUM_SURROUNDING_QUEEN

	// How many pieces can move. Two numbers per insect: the first is considering any pieces,
	// the second discards the pieces that are surrounding the opponent's queen (and presumably
	// not to be moved)
	F_NUM_CAN_MOVE
	F_OPP_NUM_CAN_MOVE

	// Number of moves threatening to reach around opponents queen.
	// Two counts here: the first is the number of pieces that can
	// reach around the opponent's queen. The second is the number
	// of free positions around the opponent's queen that can be
	// reached.
	F_NUM_THREATENING_MOVES
	F_OPP_NUM_THREATENING_MOVES

	// Number of moves till a draw due to running out of moves. Max to 10.
	F_MOVES_TO_DRAW

	// Number of pieces that are "leaves" (only one neighbor)
	// First number is for current player, the second is for the
	// opponent.
	F_NUM_SINGLE

	// Whether there is an opponent BEETLE on top of QUEEN.
	F_QUEEN_COVERED

	// Average manhattan distance to opposing queen for each of the piece types.
	F_AVERAGE_DISTANCE_TO_QUEEN
	F_OPP_AVERAGE_DISTANCE_TO_QUEEN

	// Last entry.
	F_NUM_FEATURES
)

// FeatureDef includes feature name, dimension and index in the concatenation of features.
type FeatureDef struct {
	FId  FeatureId
	Name string
	Dim  int

	// VecIndex refers to the index in the concatenated feature vector.
	VecIndex int
	Setter   FeatureSetter

	// Number of feature (AllFeaturesDim) when this feature was created.
	Version int
}

var (
	// Enumeration, in order, of the features extracted by FeatureVector.
	// The VecIndex attribute is properly set during the package initialization.
	// The  "Opp" prefix refers to opponent.
	AllFeatures = [F_NUM_FEATURES]FeatureDef{
		{F_NUM_OFFBOARD, "NumOffboard", int(NumPieceTypes), 0, fNumOffBoard, 0},
		{F_OPP_NUM_OFFBOARD, "OppNumOffboard", int(NumPieceTypes), 0, fNumOffBoard, 0},

		{F_NUM_SURROUNDING_QUEEN, "NumSurroundingQueen", 1, 0, fNumSurroundingQueen, 0},
		{F_OPP_NUM_SURROUNDING_QUEEN, "OppNumSurroundingQueen", 1, 0, fNumSurroundingQueen, 0},

		{F_NUM_CAN_MOVE, "NumCanMove", 2 * int(NumPieceTypes), 0, fNumCanMove, 0},
		{F_OPP_NUM_CAN_MOVE, "OppNumCanMove", 2 * int(NumPieceTypes), 0, fNumCanMove, 0},

		{F_NUM_THREATENING_MOVES, "NumThreateningMoves", 2, 0, fNumThreateningMoves, 0},
		{F_OPP_NUM_THREATENING_MOVES, "OppNumThreateningMoves", 2, 0, fNumThreateningMoves, 39},

		{F_MOVES_TO_DRAW, "MovesToDraw", 1, 0, fNumToDraw, 0},
		{F_NUM_SINGLE, "NumSingle", 2, 0, fNumSingle, 0},
		{F_QUEEN_COVERED, "QueenIsCovered", 2, 0, fQueenIsCovered, 41},
		{F_AVERAGE_DISTANCE_TO_QUEEN, "AverageDistanceToQueen",
			int(NumPieceTypes), 0, fAverageDistanceToQueen, 51},
		{F_OPP_AVERAGE_DISTANCE_TO_QUEEN, "OppAverageDistanceToQueen",
			int(NumPieceTypes), 0, fAverageDistanceToQueen, 51},
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

// LabeledExample can be used for training.
type LabeledExample struct {
	Features []float32
	Label    float32

	ActionsFeatures [][]float32 // Optional
	ActionLabels    [][]float32
}

func MakeLabeledExample(board *Board, label float32, version int) LabeledExample {
	return LabeledExample{
		FeatureVector(board, version), label, nil, nil}
}

// FeatureVector calculates the feature vector, of length AllFeaturesDim, for the given
// board.
// Models created at different times may use different subsets of features. This is
// specified by providing the number of features expected by the model.
func FeatureVector(b *Board, version int) (f []float32) {
	if version > AllFeaturesDim {
		log.Panicf("Requested %d features, but only know about %d", version, AllFeaturesDim)
	}
	f = make([]float32, AllFeaturesDim)
	for ii := range AllFeatures {
		featDef := &AllFeatures[ii]
		if featDef.Version <= version {
			featDef.Setter(b, featDef, f)
		}
	}

	if version != AllFeaturesDim {
		// Filter only features for given version.
		newF := make([]float32, 0, version)
		for ii := range AllFeatures {
			featDef := &AllFeatures[ii]
			if featDef.Version <= version {
				newF = append(newF, f[featDef.VecIndex:featDef.VecIndex+featDef.Dim]...)
			}
		}
		f = newF
	}

	return
}

func PrettyPrintFeatures(f []float32) {
	for ii := range AllFeatures {
		def := &AllFeatures[ii]
		fmt.Printf("\t%s: ", def.Name)
		if def.Dim == 1 {
			fmt.Printf("%.2f", f[def.VecIndex])
		} else {
			fmt.Printf("%v", f[def.VecIndex:def.VecIndex+def.Dim])
		}
		fmt.Println()
	}
}

func fNumOffBoard(b *Board, def *FeatureDef, f []float32) {
	idx := def.VecIndex
	player := b.NextPlayer
	if def.FId == F_OPP_NUM_OFFBOARD {
		player = b.OpponentPlayer()
	}
	for _, piece := range Pieces {
		f[idx+int(piece)-1] = float32(b.Available(player, piece))
	}
}

func fNumSurroundingQueen(b *Board, def *FeatureDef, f []float32) {
	idx := def.VecIndex
	player := b.NextPlayer
	if def.FId == F_OPP_NUM_SURROUNDING_QUEEN {
		player = b.OpponentPlayer()
	}
	f[idx] = float32(b.Derived.NumSurroundingQueen[player])
}

func fNumCanMove(b *Board, def *FeatureDef, f []float32) {
	idx := def.VecIndex
	player := b.NextPlayer
	opponent := b.OpponentPlayer()
	if def.FId == F_OPP_NUM_CAN_MOVE {
		player, opponent = opponent, player
	}
	actions := b.Derived.PlayersActions[player]
	var queenNeighbours []Pos
	if b.Available(opponent, QUEEN) == 0 {
		queenNeighbours = b.OccupiedNeighbours(b.Derived.QueenPos[opponent])
	}

	counts := make(map[PieceType]int)
	countsNotQueenNeighbours := make(map[PieceType]int)
	posVisited := make(map[Pos]bool)
	for _, action := range actions {
		if action.Move && !posVisited[action.SourcePos] {
			posVisited[action.SourcePos] = true
			counts[action.Piece]++
			if !posInSlice(queenNeighbours, action.SourcePos) {
				countsNotQueenNeighbours[action.Piece]++
			}
		}
	}
	for _, piece := range Pieces {
		f[idx+2*(int(piece)-1)] = float32(counts[piece])
		f[idx+1+2*(int(piece)-1)] = float32(countsNotQueenNeighbours[piece])
	}
}

func posInSlice(slice []Pos, p Pos) bool {
	for _, sPos := range slice {
		if p == sPos {
			return true
		}
	}
	return false
}

func fNumThreateningMoves(b *Board, def *FeatureDef, f []float32) {
	idx := def.VecIndex
	player := b.NextPlayer
	opponent := b.OpponentPlayer()
	if def.FId == F_OPP_NUM_CAN_MOVE {
		player, opponent = opponent, player
	}
	actions := b.Derived.PlayersActions[player]
	f[idx] = 0
	f[idx+1] = 0
	if b.Available(opponent, QUEEN) > 0 {
		// Queen not yet set up.
		return
	}

	// Add
	freeOppQueenNeighbors := b.Derived.QueenPos[opponent].Neighbours()
	usedPieces := make(map[Pos]bool)
	usedPositions := make([]Pos, 0, len(freeOppQueenNeighbors))
	canPlaceAroundQueen := false
	for _, action := range actions {
		if !posInSlice(freeOppQueenNeighbors, action.TargetPos) ||
			posInSlice(freeOppQueenNeighbors, action.SourcePos) {
			continue
		}
		if action.Move {
			if !usedPieces[action.SourcePos] {
				// Number of pieces that can reach around opponent's queen.
				usedPieces[action.SourcePos] = true
				f[idx]++
			}
		} else {
			// Placement can happen when there is a bettle on top of the
			// opponent Queen.
			canPlaceAroundQueen = true
			continue
		}
		if !posInSlice(usedPositions, action.TargetPos) {
			// Number of positions around opponent's queen that can be reached.
			usedPositions = append(usedPositions, action.TargetPos)
			f[idx+1]++
		}
	}
	if canPlaceAroundQueen {
		// In this case any of the available pieces for placement can
		// be put around the Queen.
		f[idx] += float32(TotalPiecesPerPlayer - b.Derived.NumPiecesOnBoard[player])
	}
}

func fNumToDraw(b *Board, def *FeatureDef, f []float32) {
	idx := def.VecIndex
	f[idx] = float32(b.MaxMoves - b.MoveNumber + 1)
	if f[idx] > 10 {
		f[idx] = 10
	}

}

func fNumSingle(b *Board, def *FeatureDef, f []float32) {
	idx := def.VecIndex
	player := b.NextPlayer
	opponent := b.OpponentPlayer()
	f[idx] = float32(b.Derived.Singles[player])
	f[idx+1] = float32(b.Derived.Singles[opponent])
}

func fQueenIsCovered(b *Board, def *FeatureDef, f []float32) {
	idx := def.VecIndex
	player := b.NextPlayer
	opponent := b.OpponentPlayer()
	for ii := 0; ii < 2; ii++ {
		pos := b.Derived.QueenPos[player]
		posPlayer, _, _ := b.PieceAt(pos)
		if posPlayer != player {
			f[idx+ii] = 1.0
		} else {
			f[idx+ii] = 0.0
		}
		// Invert players selection.
		player, opponent = opponent, player
	}
}

// Calculates the average manhattan distance from each piece type to the opponent queen.
// Pieces not in place will count as the furthest piece + 1.
func fAverageDistanceToQueen(b *Board, def *FeatureDef, f []float32) {
	idx := def.VecIndex
	player := b.NextPlayer
	opponent := b.OpponentPlayer()
	if def.FId == F_OPP_AVERAGE_DISTANCE_TO_QUEEN {
		player, opponent = opponent, player
	}
	queenPos := b.Derived.QueenPos[opponent]
	var totals [NumPieceTypes]int
	var maxDist int
	b.EnumeratePieces(func(pPlayer uint8, piece PieceType, pos Pos, covered bool) {
		if pPlayer != player {
			return
		}
		dist := queenPos.Distance(pos)
		if dist > maxDist {
			maxDist = dist
		}
		totals[piece-1] += dist
	})
	maxDist++ // Distance of non-placed pieces are set to maxDist+1.
	for _, piece := range Pieces {
		totals[piece-1] += (maxDist * int(b.Available(player, piece)))
	}
	for ii := 0; ii < int(NumPieceTypes); ii++ {
		f[idx+ii] = float32(totals[ii]) / float32(InitialAvailability[ii])
	}
	return

}

// fullBoardDimensions returns the minimal dimensions
// required to fit the board.
func FullBoardDimensions(b *Board) (width, height int) {
	width, height = b.Width()+2, b.Height()+2
	if width%2 != 0 {
		width++
	}
	return
}

const (
	SuggestedFullBoardWidth  = 24
	SuggestedFullBoardHeight = 24
)

var EmptyCellRow = make([][]float32, SuggestedFullBoardWidth)

func init() {
	for col := range EmptyCellRow {
		EmptyCellRow[col] = EmptyCellFeatures
	}
}

// MakeFullBoardFeatures returns features for the full board within
// an area of height/width. It will panic if the area is not able to
// contain the current board state -- use FullBoardDimensions.
// Empty hexagons will be filled with zeroes.
//
// It returns a multi-D vector of shape `[height][width][FEATURES_PER_POSITION]`
// and the shift of X and Y from the original map:
//
//	original_x + shift_x = FullBoardFeatures_x
//	original_y + shift_y = FullBoardFeatures_y
//
// Notice that shift_x will be even, so that the parity of the
// hexagonal map remains constant -- the value of x%2 affects the neighbourhood
// in the grid.
func MakeFullBoardFeatures(b *Board, width, height int) (features [][][]float32) {
	minWidth, minHeight := FullBoardDimensions(b)
	if width < minWidth || height < minHeight {
		klog.Fatalf("FullBoardFeatures for board of size (%d, %d) not possible on reserved space (%d, %d)",
			b.Height(), b.Width(), height, width)
	}

	features = make([][][]float32, height)
	shiftX, shiftY := FullBoardShift(b)
	for fbY := range features {
		if fbY > minHeight+1 && width == SuggestedFullBoardWidth {
			// Use
			features[fbY] = EmptyCellRow
		} else {
			features[fbY] = make([][]float32, width)
			row := features[fbY]
			for fbX := range row {
				pos := Pos{int8(fbX + shiftX), int8(fbY + shiftY)}
				row[fbX] = PositionFeatures(b, pos)
			}
		}
	}
	return
}

func FullBoardShift(b *Board) (shiftX, shiftY int) {
	shiftX = int(b.Derived.MinX) - 1
	shiftY = int(b.Derived.MinY) - 1
	if shiftX%2 != 0 {
		shiftX--
	}
	return
}

func PosToFullBoardPosition(b *Board, pos Pos) [2]int64 {
	shiftX, shiftY := FullBoardShift(b)
	return [2]int64{int64(int(pos.X()) - shiftX), int64(int(pos.Y()) - shiftY)}
}
