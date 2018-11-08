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
//     Also the model should be invariant to rotation. So during training one could
//     augment the data with random rotations.
//  Total: 1 + 6 + 12 = 19 Position information.
//
// Information per position in the Radius-2 Area:
//     * Player owner of the piece (1): -1 is player 0, 0 if the position is empty, +1 for player 1
//     * One hot encoding of the piece in (the bottom of) the location. (6)
//     * (Target Position Only): Whether this was the source position (1)
//     * Stacked beatle pieces, from TOP to BOTTOM:
//       * Up to 4 player owner of the beatle piece (+1/-1) or zero if nothing stacked. (4)
//  Total dimension per position: 12 (for target position) or 11 (for source position)

package ai

// The 18 radius-2 neighbour positions (given as delta-X and delta-Y), aggregated
// in groups of 3 that if rotated should be equivalent.
//
// There is the version for x%2=0 (EVEN), and X%2=1 (ODD).
//
// TODO: generate this dynamically during initialization, if some day we want radius > 2.
var (
	X_EVEN_NEIGHBOURS = [6][3][2]int8{
		{{0, -1}, {0, -2}, {1, -2}},
		{{1, -1}, {2, -1}, {2, 0}},
		{{1, 0}, {2, 1}, {1, 1}},
		{{0, 1}, {0, 2}, {-1, 1}},
		{{-1, 0}, {-2, 1}, {-2, 0}},
		{{-1, -1}, {-2, -1}, {-1, -2}},
	}

	X_ODD_NEIGHBOURS = [6][3][2]int8{
		{{0, -1}, {0, -2}, {1, -1}},
		{{1, 0}, {2, -1}, {2, 0}},
		{{1, 1}, {2, 1}, {1, 2}},
		{{0, 1}, {0, 2}, {-1, 2}},
		{{-1, 1}, {-2, 1}, {-2, 0}},
		{{-1, 0}, {-2, -1}, {-1, -1}},
	}
)
