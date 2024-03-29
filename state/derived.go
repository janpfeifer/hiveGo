// Package state holds information about a game state.
//
// This file holds the function that orchestrates the building of information
// derived from the game state: valid actions, useful information (features) for
// good game play, useful information for the UI, etc.
package state

import (
	"fmt"
	"log"
	"math/rand"
)

var _ = fmt.Printf

// Maximum number of times a board may be repeated before a draw is issued.
const MAX_BOARD_REPEATS = 3

// Derived holds information that is generated from the Board state.
type Derived struct {
	// Number of times this exact same board has been seen earlier in the Match.
	Repeats uint8

	// Limits of the pieces on board.
	MinX, MaxX, MinY, MaxY int8

	// Hash of the Board. Usually unique, but not guaranteed.
	// One of the things that the Hash doesn't cover if the previous state
	// in the same match.
	Hash uint64

	// Normalized (shifted) and sorted list of positions occupied. Used by hash
	// calculation.
	NormalizedPosStackSlice PosStackSlice

	// Information about both players.
	NumPiecesOnBoard    [NUM_PLAYERS]uint8
	NumSurroundingQueen [NUM_PLAYERS]uint8
	PlacementPositions  [NUM_PLAYERS]map[Pos]bool
	Wins                [NUM_PLAYERS]bool  // If both players win, it is a draw.
	QueenPos            [NUM_PLAYERS]Pos   // Only valid if queen is actually in the board.
	Singles             [NUM_PLAYERS]uint8 // Count pieces that are at the tip (only one neighbour)

	// Pieces that can be removed without breaking the hive.
	RemovablePieces map[Pos]bool
	PlayersActions  [NUM_PLAYERS][]Action

	// Actions of the next player to move (shortcut to PlayersActions[NextPlayer]).
	Actions []Action

	// Player moves to end of match. Information is only available after the end of the
	// match, and needs to be back-filled. It is used for learning only.
	PlayerMovesToEnd int8
}

// Action describe a placement or a move. If `piece` is given, `posSource` can be
// ignored, since it's a placement of a new piece action (as opposed to a move)
type Action struct {
	// If not Move, it's a placement action.
	Move                 bool
	Piece                Piece
	SourcePos, TargetPos Pos
}

var SKIP_ACTION = Action{Piece: NO_PIECE}

func (a Action) IsSkipAction() bool {
	return a.Piece == NO_PIECE
}

func (a Action) String() string {
	if a.IsSkipAction() {
		return "SkipAction"
	}
	if a.Move {
		return fmt.Sprintf("Move %s: %s->%s", PieceLetters[a.Piece], a.SourcePos, a.TargetPos)
	} else {
		return fmt.Sprintf("Place %s in %s", PieceLetters[a.Piece], a.TargetPos)
	}
}

// Comparison of values.
func (a Action) Equal(a2 Action) bool {
	if a.Piece != a2.Piece {
		return false
	}
	if a.Piece == NO_PIECE {
		// NO-OP move.
		return true
	}
	if a.Move != a2.Move || !a.TargetPos.Equal(a2.TargetPos) {
		return false
	}
	if !a.Move {
		return true
	}
	return a.SourcePos == a2.SourcePos
}

// BuildDerived rebuilds information derived from the board.
func (b *Board) BuildDerived() {
	// Reset Derived.
	b.Derived = nil
	derived := &Derived{}

	// Get uncached (from Derived) results.
	derived.MinX, derived.MaxX, derived.MinY, derived.MaxY = b.UsedLimits()

	// Set new derived object.
	b.Derived = derived

	// Normalized list of positions.
	derived.NormalizedPosStackSlice = b.normalizedPosStackSlice()
	derived.Hash = b.normalizedHash()
	derived.Repeats = b.FindRepeats()

	// Per player info.
	for p := uint8(0); p < NUM_PLAYERS; p++ {
		derived.NumPiecesOnBoard[p] = TOTAL_PIECES_PER_PLAYER - b.available[p].Count()
		derived.PlacementPositions[p] = b.placementPositions(p)
	}

	derived.RemovablePieces = b.removable()
	for p := uint8(0); p < NUM_PLAYERS; p++ {
		derived.PlayersActions[p] = b.ValidActions(p)
		shuffleActions(derived.PlayersActions[p])
	}

	derived.Actions = derived.PlayersActions[b.NextPlayer]
	derived.Wins, derived.NumSurroundingQueen, derived.QueenPos = b.endGame()
	derived.Singles = b.ListSingles()
}

func shuffleActions(actions []Action) {
	for ii := range actions {
		jj := rand.Intn(len(actions))
		actions[ii], actions[jj] = actions[jj], actions[ii]
	}
}

// ValidActions returns the list of valid actions for given player.
// For the NextPlayer the list of actions is pre-cached in Derived.
func (b *Board) ValidActions(player uint8) (actions []Action) {
	actions = make([]Action, 0, 25)
	actions = b.addPlacementActions(player, actions)
	actions = b.addMoveActions(player, actions)
	return
}

// FindAction finds the index to the given action. It assumes the action is the exact same slice,
// that is, it is a shallow comparison.
func (b *Board) FindAction(action Action) int {
	for ii, action2 := range b.Derived.Actions {
		if action == action2 {
			return ii
		}
	}
	log.Panicf("Action %s chosen not found. Available: %v", action, b.Derived.Actions)
	return -1
}

// FindActionDeep like FindAction finds the index to the given action. But it does a deep-comparison, so
// the action may have been generated separatedly from the actions of the board -- for instance when loading
// a match.
func (b *Board) FindActionDeep(action Action) int {
	for ii, action2 := range b.Derived.Actions {
		if action.Equal(action2) {
			return ii
		}
	}
	log.Panicf("Action %s chosen not found. Available: %v", action, b.Derived.Actions)
	return -1
}

// placementPositions enumerate placement positions.
func (b *Board) placementPositions(player uint8) (placements map[Pos]bool) {
	placements = make(map[Pos]bool)
	if len(b.board) == 0 {
		placements[Pos{0, 0}] = true
		return
	}
	if len(b.board) == 1 && b.CountAt(Pos{0, 0}) == 1 {
		for _, pos := range (Pos{0, 0}.Neighbours()) {
			placements[pos] = true
		}
		return
	}

	// Enumerate all empty positions next to friendly pieces.
	candidates := make(map[Pos]bool)
	for pos, stacked := range b.board {
		posPlayer, _ := stacked.Top()
		if posPlayer != player {
			continue
		}
		for _, nPos := range b.EmptyNeighbours(pos) {
			candidates[nPos] = true
		}
	}

	// Filter those down to only those that have no opponent neighbours.
	for pos, _ := range candidates {
		if len(b.OpponentNeighbours(pos)) == 0 {
			placements[pos] = true
		}
	}
	return
}

// addPlacementActions adds valid placement actions to the given
// actions slice.
func (b *Board) addPlacementActions(player uint8, actions []Action) []Action {
	derived := b.Derived
	mustPlaceQueen := b.Available(player, QUEEN) > 0 && derived.NumPiecesOnBoard[player] >= 3

	for pos, _ := range derived.PlacementPositions[player] {
		if mustPlaceQueen {
			actions = append(actions, Action{Move: false, Piece: QUEEN, TargetPos: pos})
		} else {
			for _, piece := range Pieces {
				if b.Available(player, piece) > 0 {
					actions = append(actions, Action{Move: false, Piece: piece, TargetPos: pos})
				}
			}
		}
	}
	return actions
}

// IsRemovable determines whether a piece on specified position is
// removable without splitting the hive.
//
// Deprecated, use Derived.RemovablePieces instead, which is much faster.
func (b *Board) IsRemovable(pos Pos) bool {
	_, _, stacked := b.PieceAt(pos)
	if stacked {
		// If there is a piece underneath, the top piece is always removable.
		return true
	}

	neighbours := b.OccupiedNeighbours(pos)

	// If it is the first piece in game, it can't be removed.
	if len(neighbours) == 0 {
		return false
	}

	// Only one neighbor; and 5 or 6 neighbors, in which case they will forceably
	// stay connected.
	if len(neighbours) == 1 || len(neighbours) >= 5 {
		return true
	}

	// Do a DFS from one neighbor and make sure to reach the other ones. This is O(N), and since
	// IsRemovable is called for every position, overal it is a O(N^2) algorithm. Maybe there
	// is a better way to do this, leveraging the result of IsRemovable of other nodes ?
	start := neighbours[0]
	visitedMap := map[Pos]bool{pos: true}
	mustFind := make(map[Pos]bool)
	for i := 1; i < len(neighbours); i++ {
		mustFind[neighbours[i]] = true
	}
	queue := []Pos{start}
	for len(queue) > 0 {
		newQueue := []Pos{}
		for _, visiting := range queue {
			for _, nPos := range b.OccupiedNeighbours(visiting) {
				// Skip neighbours already visited.
				if _, visited := visitedMap[nPos]; visited {
					continue
				}
				visitedMap[nPos] = true

				// Mark if found path to original neighbours (mustFind).
				if _, found := mustFind[nPos]; found {
					delete(mustFind, nPos)
					if len(mustFind) == 0 {
						// All original neighboors can be reached, hence pos can be
						// removed without breaking the hive (bipartioning the graph)
						return true
					}
				}

				// Add new neighbor for visiting.
				newQueue = append(newQueue, nPos)
			}
		}
		queue = newQueue
	}

	// Not all neighbours were found, hence removing piece would bi-partition graph.
	return false
}

// addMoveActions add valid move actions to the given actions slice
func (b *Board) addMoveActions(player uint8, actions []Action) []Action {
	if b.Available(player, QUEEN) != 0 {
		// Queen not yet in the game, can't move.
		return actions
	}

	d := b.Derived
	for srcPos, piecesStack := range b.board {
		piecePlayer, piece := piecesStack.Top()
		if player != piecePlayer {
			// We are only interested in the current player.
			continue
		}
		if !d.RemovablePieces[srcPos] {
			// Skip pieces that if removed would break the hive.
			continue
		}

		var tgtPoss []Pos
		switch {
		case piece == QUEEN:
			tgtPoss = b.queenMoves(srcPos)
		case piece == SPIDER:
			tgtPoss = b.spiderMoves(srcPos)
		case piece == GRASSHOPPER:
			tgtPoss = b.grasshopperMoves(srcPos)
		case piece == ANT:
			tgtPoss = b.antMoves(srcPos)
		case piece == BEETLE:
			tgtPoss = b.beetleMoves(srcPos)
		}

		// Collect target positions into actions.
		for _, tgtPos := range tgtPoss {
			actions = append(actions, Action{Move: true, Piece: piece, SourcePos: srcPos, TargetPos: tgtPos})
		}
	}
	return actions
}

// Act takes the given action for the b.NextPlayer player and returns a new board (with
// board.Derived cleared)
//
// It DOES NOT CHECK that the action is valid (it can be useful for testing),
// and leaves that to the UI to handle.
//
// If Piece = NO_PIECE, it's assumed to be a pass-action.
//
// It also updates the derived information by calling `BuildDerived()`.
func (b *Board) Act(action Action) (newB *Board) {
	newB = b.Copy()
	if action.Piece != NO_PIECE {
		if !action.Move {
			// Placement
			newB.StackPiece(action.TargetPos, newB.NextPlayer, action.Piece)
			newB.SetAvailable(newB.NextPlayer, action.Piece,
				newB.Available(newB.NextPlayer, action.Piece)-1)
		} else {
			player, piece := newB.PopPiece(action.SourcePos)
			newB.StackPiece(action.TargetPos, player, piece)
		}
	}
	newB.NextPlayer = 1 - newB.NextPlayer
	newB.MoveNumber++
	newB.BuildDerived()
	return
}

// IsValid if given action is listed as a valid one.
func (b *Board) IsValid(action Action) bool {
	for _, validAction := range b.Derived.Actions {
		if action == validAction {
			return true
		}
	}
	return false
}

// endGame checks for end games and will return true for each of the players if they
// managed to surround the opponents queen. Returns also the number of pieces surrounding
// each queen.
func (b *Board) endGame() (wins [NUM_PLAYERS]bool, surrounding [NUM_PLAYERS]uint8, queenPos [NUM_PLAYERS]Pos) {
	if b.MoveNumber > b.MaxMoves {
		// After MaxMoves is reached, the game is considered a draw.
		wins = [NUM_PLAYERS]bool{true, true}
		return
	}
	wins = [NUM_PLAYERS]bool{false, false}
	for pos, stack := range b.board {
		if isQueen, player := stack.HasQueen(); isQueen {
			// Convert player to "NextPlayer"/"Opponent"
			queenPos[player] = pos
			surrounding[player] = uint8(len(b.OccupiedNeighbours(pos)))
			if surrounding[player] == 6 {
				// If player's queen is sorrounded, other player wins (or draws).
				wins[1-player] = true
			}
		}
	}
	return
}

func (b *Board) NumActions() int {
	return len(b.Derived.Actions)
}

func (b *Board) IsFinished() bool {
	return b.Derived.Repeats >= MAX_BOARD_REPEATS || b.Derived.Wins[0] || b.Derived.Wins[1]
}

func (b *Board) Draw() bool {
	return b.IsFinished() && (b.Derived.Repeats >= MAX_BOARD_REPEATS  ||
		b.Derived.Wins[0] == b.Derived.Wins[1])
}

func (b *Board) Winner() uint8 {
	if b.Derived.Wins[0] {
		return 0
	} else {
		return 1
	}
}

func (b *Board) ListSingles() (singles [2]uint8) {
	for pos, stack := range b.board {
		if len(b.OccupiedNeighbours(pos)) == 1 {
			player, _ := stack.Top()
			singles[player]++
		}
	}
	return
}

func (b *Board) Width() int {
	return int(b.Derived.MaxX - b.Derived.MinX)
}

func (b *Board) Height() int {
	return int(b.Derived.MaxY - b.Derived.MinY)
}

func (b* Board) EnumeratePieces(cb func (player uint8, piece Piece, pos Pos, covered bool)) {
	for pos, stack := range b.board {
		covered := false
		for stack != 0 {
			var player uint8
			var piece Piece
			stack, player, piece = stack.PopPiece()
			cb(player, piece, pos, covered)
			covered = true
		}
	}
}