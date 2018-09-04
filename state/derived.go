// Package state holds information about a game state.
//
// This file holds the function that orchestrates the building of information
// derived from the game state: valid actions, useful information (features) for
// good game play, useful information for the UI, etc.
package state

import "fmt"

var _ = fmt.Printf

// Derived holds information that is generated from the Board state.
type Derived struct {
	// Information about both players.
	NumPiecesOnBoard    [2]uint8
	NumPiecesFreeToMove [2]uint8

	// Information only about the next player to move (NextPlayer)
	Wins               [2]bool // If both players win, it is a draw.
	PlacementPositions map[Pos]bool
	RemovablePieces    map[Pos]bool
	Actions            []Action
}

// Action describe a placement or a move. If `piece` is given, `posSource` can be
// ignored, since it's a placement of a new piece action (as opposed to a move)
type Action struct {
	// If not Move, it's a placement action.
	Move                 bool
	Piece                Piece
	SourcePos, TargetPos Pos
}

// BuildDerived rebuilds information derived from the board.
func (b *Board) BuildDerived() {
	b.Derived = &Derived{}
	derived := b.Derived

	// Count of pieces.
	for p := 0; p < 2; p++ {
		derived.NumPiecesOnBoard[p] = TOTAL_PIECES_PER_PLAYER - b.available[p].Count()
	}

	derived.PlacementPositions = b.placementPositions()
	derived.RemovablePieces = make(map[Pos]bool)
	for pos, _ := range b.board {
		if b.IsRemovable(pos) {
			derived.RemovablePieces[pos] = true
		}
	}
	derived.Actions = b.validActions()
	derived.Wins = b.endGame()
}

// ValidActions returns the list of valid actions for the NextPlayer.
func (b *Board) validActions() (actions []Action) {
	actions = make([]Action, 0, 25)
	actions = b.addPlacementActions(actions)
	actions = b.addMoveActions(actions)
	return
}

// placementPositions enumerate placement positions.
func (b *Board) placementPositions() (placements map[Pos]bool) {
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
		player, _ := stacked.Top()
		if player != b.NextPlayer {
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
func (b *Board) addPlacementActions(actions []Action) []Action {
	player := b.NextPlayer
	derived := b.Derived
	mustPlaceQueen := b.Available(player, QUEEN) > 0 && derived.NumPiecesOnBoard[player] >= 3

	for pos, _ := range derived.PlacementPositions {
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

// Determine whether a piece on specified position is removable without
// splitting the hive.
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
func (b *Board) addMoveActions(actions []Action) []Action {
	d := b.Derived
	for srcPos, piecesStack := range b.board {
		player, piece := piecesStack.Top()
		if player != b.NextPlayer {
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
// managed to sorround the opponents queen.
func (b *Board) endGame() (wins [2]bool) {
	if b.MoveNumber > b.MaxMoves {
		// After MaxMoves is reached, the game is considered a draw.
		wins = [2]bool{true, true}
		return
	}
	wins = [2]bool{false, false}
	for pos, stack := range b.board {
		if isQueen, player := stack.HasQueen(); isQueen {
			if len(b.OccupiedNeighbours(pos)) == 6 {
				// If player's queen is sorrounded, other player wins (or draws).
				wins[1-player] = true
			}
		}
	}
	return
}
