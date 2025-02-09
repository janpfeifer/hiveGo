package features

// Policy features add a set of features per action of the board, plus padding.
//
// Each action is represented by:
//
//   Move: 0/1 if the action is a move (and not a new placement).
//   Source Position: (if move, otherwise 0)
//     Radius-2 of board area around piece leaving.
//   Target Position:
//     Radius-2 of board area around target position. This
//     radius-2 area will include the new piece location, and will
//     have the piece removed from where it was, if it was a move
//     (and not a new placement), and it was in the same radius-2
//     area. This radius-2 area will include a bit indicating
//     where the piece came from, Only set if the move is within
//     the same radius-2 area.
//
// Radius-2 Area: (Both for Source Position and Target Position)
//     Includes the center position (1), plus the immediate neighbours (6),
//     plus the neighbours of the neighbours (12), so it will be a concatenation
//     of the description of 19 positions.
//     Notice that one has to be careful with finding the neighbours: the mapping
//     around an (X,Y) position depends on the parity of the X coordinate (X%2).
//     Also the model should be invariant to rotation: this is achieved in TensorFlow
//     by rotating in the 6 different positions and doing a pooling on those.
//  Total: 1 + 6 + 12 = 19 Position information.
//
// See below in the constants defined for "Features Per Position" for a
// description of the information per position in the Radius-2 area.

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
)

const (
	POSITIONS_PER_SECTION = 3 // Value for radius-2
)

// The 18 radius-2 neighbour positions (given as delta-X and delta-Y), aggregated
// in groups of 3 that if rotated should be equivalent.
//
// There is the version for x%2=0 (EVEN), and X%2!=0 (ODD).
//
// TODO: generate this dynamically during initialization, if some day we want radius > 2.
var (
	X_EVEN_NEIGHBOURS = [6][POSITIONS_PER_SECTION][2]int8{
		{{0, -1}, {0, -2}, {1, -2}},
		{{1, -1}, {2, -1}, {2, 0}},
		{{1, 0}, {2, 1}, {1, 1}},
		{{0, 1}, {0, 2}, {-1, 1}},
		{{-1, 0}, {-2, 1}, {-2, 0}},
		{{-1, -1}, {-2, -1}, {-1, -2}},
	}

	X_ODD_NEIGHBOURS = [6][POSITIONS_PER_SECTION][2]int8{
		{{0, -1}, {0, -2}, {1, -1}},
		{{1, 0}, {2, -1}, {2, 0}},
		{{1, 1}, {2, 1}, {1, 2}},
		{{0, 1}, {0, 2}, {-1, 2}},
		{{-1, 1}, {-2, 1}, {-2, 0}},
		{{-1, 0}, {-2, -1}, {-1, -1}},
	}
)

type ActionPositionFeatures struct {
	// Features related to the center of te
	Center []float32

	// Sections should be invariant to rotation.
	Sections [6][]float32
}

type ActionFeatures struct {
	Move                           float32
	SourceFeatures, TargetFeatures ActionPositionFeatures
}

// ActionFeatures build the features for one action. We do this one at a time so that
// they can be accumulated directly into a tensor (or whatever is the backend machine
// learning)
func NewActionFeatures(b *Board, action Action, policyVersion int) (af ActionFeatures) {
	if action.Move {
		af.Move = 1
		af.SourceFeatures.neighbourhoodFeatures(b, action, policyVersion, action.SourcePos, false)
	} else {
		af.Move = 0
		// Zero out SourceFeatures.
		af.SourceFeatures.Center = make([]float32, IdPositionLast)
		for ii := 0; ii < 6; ii++ {
			af.SourceFeatures.Sections[ii] = make([]float32, POSITIONS_PER_SECTION*IdPositionLast)
		}
	}
	af.TargetFeatures.neighbourhoodFeatures(b, action, policyVersion, action.TargetPos, true)
	return
}

// newNeighbourhoodFeatures generates the features for the neighbourhood around given position.
// If exe2cAction is set to true, the action is simulated in the map (piece removed from source position,
// and placed on target position).
func (f *ActionPositionFeatures) neighbourhoodFeatures(b *Board, action Action, policyVersion int, pos Pos, execAction bool) {
	f.Center = positionActionFeatures(b, action, policyVersion, pos, execAction)
	neighbourhood := &X_EVEN_NEIGHBOURS
	if pos.X()%2 != 0 {
		neighbourhood = &X_ODD_NEIGHBOURS
	}
	for section := 0; section < 6; section++ {
		f.Sections[section] = make([]float32, 0, POSITIONS_PER_SECTION*IdPositionLast)
		for ii := 0; ii < POSITIONS_PER_SECTION; ii++ {
			neighPos := Pos{pos.X() + neighbourhood[section][ii][0], pos.Y() + neighbourhood[section][ii][1]}
			neighFeatures := positionActionFeatures(b, action, policyVersion, neighPos, execAction)
			f.Sections[section] = append(f.Sections[section], neighFeatures...)
			//if section == 3 && ii == 1 {
			//	fmt.Printf("Sec 3 #1 %s -> %s: %s\n",
			//		pos, neighPos,
			//		PositionFeaturesToString(neighFeatures))
			//}
		}
	}
}

func playerToValue(b *Board, player PlayerNum) float32 {
	if player == b.NextPlayer {
		return 1
	} else {
		return -1
	}
}

func positionActionFeatures(b *Board, action Action, policyVersion int, pos Pos, execAction bool) []float32 {
	stack := b.StackAt(pos)
	if execAction {
		// Fake execution of action.
		if action.Move && pos == action.SourcePos {
			stack, _, _ = stack.PopPiece()
		} else if pos == action.TargetPos {
			stack = stack.StackPiece(b.NextPlayer, action.Piece)
		}
	}
	return stackFeatures(b, pos, stack)
}
