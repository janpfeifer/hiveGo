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
package ai

import (
	"fmt"
	"github.com/golang/glog"

	. "github.com/janpfeifer/hiveGo/state"
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
		af.SourceFeatures.Center = make([]float32, FEATURES_PER_POSITION)
		for ii := 0; ii < 6; ii++ {
			af.SourceFeatures.Sections[ii] = make([]float32, POSITIONS_PER_SECTION*FEATURES_PER_POSITION)
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
		f.Sections[section] = make([]float32, 0, POSITIONS_PER_SECTION*FEATURES_PER_POSITION)
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

// Features Per Position:
const (
	// +1 for current player, -1 for opponent player or 0 if there are no pieces
	// at this position.
	POS_FEATURE_PLAYER_OWNER = 0

	// This is the one-hot encoding of the piece on the top of the stack.
	POS_FEATURE_PIECE_ONE_HOT = POS_FEATURE_PLAYER_OWNER + 1

	// Whether the piece is removable without breaking the hive.
	POS_FEATURE_IS_PIECE_REMOVABLE = POS_FEATURE_PIECE_ONE_HOT + int(NUM_PIECE_TYPES)

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
	FEATURES_PER_POSITION = POS_FEATURE_STACK_BOTTOM_PIECE_ONE_HOT + int(NUM_PIECE_TYPES)
)

func playerToValue(b *Board, player uint8) float32 {
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

func PositionFeatures(b *Board, pos Pos) []float32 {
	return stackFeatures(b, pos, b.StackAt(pos))
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
		// Piece at very bottom is stored separately, in the next field:
		f[POS_FEATURE_STACKED_BEATLE_OWNER+stackPos-1] = 0

		// Information about piece at the very bottom of stack -- in most cases the same as the top of the stack.
		f[POS_FEATURE_STACK_BOTTOM_PIECE_OWNER] = playerToValue(b, player)
		f[POS_FEATURE_STACK_BOTTOM_PIECE_ONE_HOT+int(piece-1)] = 1

		// Is piece removable?
		if b.Derived.RemovablePieces[pos] {
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
	for ii := ANT; ii < LAST_PIECE_TYPE; ii++ {
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
		glog.Error(msg)
		return msg
	}

	return fmt.Sprintf("[Top: %s, Stack(%v), Bottom: %s / Removable=%g]",
		pieceFeatureToStr(f[POS_FEATURE_PLAYER_OWNER:]),
		f[POS_FEATURE_STACKED_BEATLE_OWNER:POS_FEATURE_STACKED_BEATLE_OWNER+3],
		pieceFeatureToStr(f[POS_FEATURE_STACK_BOTTOM_PIECE_OWNER:]),
		f[POS_FEATURE_IS_PIECE_REMOVABLE])
}

func init() {
	glog.V(1).Infof("Number of features per position = %d\n", FEATURES_PER_POSITION)
}
