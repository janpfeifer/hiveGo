# Tools to build lattice model.
import tensorflow_lattice as tfl
import tensorflow as tf

_NUM_PIECE_TYPES=5
_TOTAL_AVAILABILITY = 11
_INITIAL_AVAILABILITY = [3, 2, 3, 1, 2]
_BOARD_FEATURES_DIM=41

# A tuple with 4 values for each feature type: min, max, num calibration points, lattice size.
_FEATURES_DEFINITIONS = []

# F_NUM_OFFBOARD: from 0 to 11
# F_OPP_NUM_OFFBOARD
for ii in range(2):
    for numPieces in _INITIAL_AVAILABILITY:
        _FEATURES_DEFINITIONS.append((0, numPieces, numPieces+1, 2))

# How many pieces are around the queen (0 if queen hasn't been placed)
# F_NUM_SURROUNDING_QUEEN
# F_OPP_NUM_SURROUNDING_QUEEN
for ii in range(2):
    _FEATURES_DEFINITIONS.append((1, 0, 5, 6))

# How many pieces can move. Two numbers per insect: the first is considering any pieces,
# the second discards the pieces that are surrounding the opponent's queen (and presumably
# not to be moved)
# F_NUM_CAN_MOVE
# F_OPP_NUM_CAN_MOVE
for ii in range(2):
    for numPieces in _INITIAL_AVAILABILITY:
        for jj in range(2):
            _FEATURES_DEFINITIONS.append((0, numPieces, numPieces+1, 2))

# Number of moves threatening to reach around opponents queen.
# Two counts here: the first is the number of pieces that can
# reach around the opponent's queen. The second is the number
# of free positions around the opponent's queen that can be
# reached.
#F_NUM_THREATENING_MOVES
#F_OPP_NUM_THREATENING_MOVES
for ii in range(2):
    for jj in range(2):
        _FEATURES_DEFINITIONS.append((0, _TOTAL_AVAILABILITY, _TOTAL_AVAILABILITY+1, 2))

# Number of moves till a draw due to running out of moves. Max to 10.
# F_MOVES_TO_DRAW
_FEATURES_DEFINITIONS.append((0, 10, 11, 2))

# Number of pieces that are "leaves" (only one neighbor)
# First number is for current player, the second is for the
# opponent.
# F_NUM_SINGLE
_FEATURES_DEFINITIONS.append((0, _TOTAL_AVAILABILITY, _TOTAL_AVAILABILITY+1, 2))
_FEATURES_DEFINITIONS.append((0, _TOTAL_AVAILABILITY, _TOTAL_AVAILABILITY+1, 2))

# Whether there is an opponent BEETLE on top of QUEEN.
# F_QUEEN_COVERED
_FEATURES_DEFINITIONS.append((0, 1, 2, 2))
_FEATURES_DEFINITIONS.append((0, 1, 2, 2))


def BuildBoardFeaturesCalibrator(board_features):
    """Use piece-wise-linear calibration to calibrate features.

    It has fixed ranges that depend on the definitions of the features in `features.go`

    Returns:
        calibrated_board_features: a tensor with the exact same shape, with values
            from 0 to 1.
        regularization_loss: a tensor with loss due to regularization of calibration.
    """
    return board_features, 0.0
