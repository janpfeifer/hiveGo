// Package state holds information about a game state.
//
// This file holds the function that orchestrates the building of information
// derived from the game state: valid actions, useful information (features) for
// good game play, useful information for the UI, etc.
package state

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"log"
	"math/rand"
)

var _ = fmt.Printf

// MaxBoardRepeats after which a draw is issued.
const MaxBoardRepeats = 3

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
	NumPiecesOnBoard    [NumPlayers]uint8
	NumSurroundingQueen [NumPlayers]uint8
	PlacementPositions  [NumPlayers]generics.Set[Pos]
	Wins                [NumPlayers]bool  // If both players win, it is a draw.
	QueenPos            [NumPlayers]Pos   // Only valid if queen is actually in the board.
	Singles             [NumPlayers]uint8 // Count pieces that are at the tip (only one neighbour)

	// Pieces that can be removed without breaking the hive.
	RemovablePositions generics.Set[Pos]
	PlayersActions     [NumPlayers][]Action

	// Actions of the next player to move (shortcut to PlayersActions[NextPlayer]).
	Actions []Action

	// nextBoards are the cached generated boards for all possible actions taken.
	// If set, it has the same length as Actions.
	//
	// It is returned by Board.TakeAllActions.
	nextBoards []*Board

	// Player moves to end of match. Information is only available after the end of the
	// match, and needs to be back-filled. It is used for learning only.
	PlayerMovesToEnd int8
}

// Action describe a placement or a move. If `piece` is given, `posSource` can be
// ignored, since it's a placement of a new piece action (as opposed to a move)
type Action struct {
	// If not Move, it's a placement action.
	Move                 bool
	Piece                PieceType
	SourcePos, TargetPos Pos
}

// SkipAction can only be played if there are no other actions to be taken.
var SkipAction = Action{Piece: NoPiece}

func (a Action) IsSkipAction() bool {
	return a.Piece == NoPiece
}

func (a Action) String() string {
	if a.IsSkipAction() {
		return "Pass (no action)"
	}
	if a.Move {
		return fmt.Sprintf("Move %s: %s->%s", PieceLetters[a.Piece], a.SourcePos, a.TargetPos)
	} else {
		return fmt.Sprintf("Place %s in %s", PieceLetters[a.Piece], a.TargetPos)
	}
}

// Equal compares whether two actions are the same.
func (a Action) Equal(a2 Action) bool {
	if a.Piece != a2.Piece {
		return false
	}
	if a.Piece == NoPiece {
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
	for p := PlayerNum(0); p < NumPlayers; p++ {
		derived.NumPiecesOnBoard[p] = TotalPiecesPerPlayer - b.available[p].Count()
		derived.PlacementPositions[p] = b.placementPositions(p)
	}

	derived.RemovablePositions = b.RemovablePositions()
	for p := PlayerNum(0); p < NumPlayers; p++ {
		derived.PlayersActions[p] = b.ValidActions(p)
		//shuffleActions(derived.PlayersActions[p])
	}

	derived.Actions = derived.PlayersActions[b.NextPlayer]
	if len(derived.Actions) == 0 {

	}
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
// If there are no valid actions, it appends the SkipAction. So there will always be a valid action.
func (b *Board) ValidActions(player PlayerNum) []Action {
	actions := make([]Action, 0, 25)
	actions = b.addPlacementActions(player, actions)
	actions = b.addMoveActions(player, actions)
	if len(actions) == 0 {
		// If there are no valid actions, add SkipAction.
		actions = append(actions, SkipAction)
	}
	return actions
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
// the action may have been generated separately from the actions of the board -- for instance when loading
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

var InitialPos = Pos{0, 0}

// placementPositions enumerate placement positions.
func (b *Board) placementPositions(player PlayerNum) (placements generics.Set[Pos]) {
	placements = make(generics.Set[Pos])
	if len(b.board) == 0 {
		// On the first move, with an empty board, only 0,0 can be used for placing.
		placements.Insert(InitialPos)
		return
	}
	if len(b.board) == 1 && b.CountAt(InitialPos) == 1 {
		for pos := range InitialPos.NeighboursIter() {
			placements.Insert(pos)
		}
		return
	}

	// Enumerate all empty positions next to friendly pieces.
	candidates := generics.MakeSet[Pos]()
	for pos, stacked := range b.board {
		posPlayer, _ := stacked.Top()
		if posPlayer != player {
			continue
		}
		for emptyPos := range b.EmptyNeighboursIter(pos) {
			candidates.Insert(emptyPos)
		}
	}

	// Filter those down to only those that have no opponent neighbours.
	for pos := range candidates.Iter() {
		hasOpponentNeighbours := false
		for neighbourPos := range b.OccupiedNeighboursIter(pos) {
			neighbourPlayer, _, _ := b.PieceAt(neighbourPos)
			if neighbourPlayer != player {
				hasOpponentNeighbours = true
				break
			}
		}
		if !hasOpponentNeighbours {
			placements.Insert(pos)
		}
	}
	return
}

// addPlacementActions adds valid placement actions to the given
// actions slice.
func (b *Board) addPlacementActions(player PlayerNum, actions []Action) []Action {
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

// oldIsRemovable determines whether a piece on specified position is
// removablePositions without splitting the hive.
//
// Deprecated, use Derived.RemovablePositions instead, which is much faster.
func (b *Board) oldIsRemovable(pos Pos) bool {
	_, _, stacked := b.PieceAt(pos)
	if stacked {
		// If there is a piece underneath, the top piece is always removablePositions.
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
	// oldIsRemovable is called for every position, overal it is a O(N^2) algorithm. Maybe there
	// is a better way to do this, leveraging the result of oldIsRemovable of other nodes ?
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
func (b *Board) addMoveActions(player PlayerNum, actions []Action) []Action {
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
		if !d.RemovablePositions.Has(srcPos) {
			// Skip pieces that if removal would break the hive.
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
// If PieceType = NoPiece, it's assumed to be a pass-action.
//
// It also updates the derived information by calling `BuildDerived()`.
func (b *Board) Act(action Action) (newB *Board) {
	newB = b.Clone()
	if action.Piece != NoPiece {
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

// TakeAllActions returns the boards generated by taking all actions available to current player.
func (b *Board) TakeAllActions() []*Board {
	if b.Derived == nil {
		b.BuildDerived()
	}
	d := b.Derived
	if d.nextBoards != nil {
		return d.nextBoards
	}
	d.nextBoards = make([]*Board, len(d.Actions))
	for actionIdx, action := range d.Actions {
		d.nextBoards[actionIdx] = b.Act(action)
	}
	return d.nextBoards
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
func (b *Board) endGame() (wins [NumPlayers]bool, surrounding [NumPlayers]uint8, queenPos [NumPlayers]Pos) {
	if b.MoveNumber > b.MaxMoves {
		// After MaxMoves is reached, the game is considered a draw.
		wins = [NumPlayers]bool{true, true}
		return
	}
	wins = [NumPlayers]bool{false, false}
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

// IsFinished returns whether the board represents a finished match.
// It depends on Derived.
func (b *Board) IsFinished() bool {
	return b.Derived.Repeats >= MaxBoardRepeats || b.Derived.Wins[0] || b.Derived.Wins[1] || b.MoveNumber >= b.MaxMoves
}

func (b *Board) FinishReason() string {
	if !b.IsFinished() {
		return "game not finished yet"
	}
	if b.Winner() != PlayerInvalid {
		return fmt.Sprintf("%s won", b.Winner())
	}
	if b.Derived.Repeats >= MaxBoardRepeats {
		return fmt.Sprintf("current board position was repeated %d time", MaxBoardRepeats)
	}
	if b.MoveNumber >= b.MaxMoves {
		return fmt.Sprintf("max number of moves %d (one per player) was reached", b.MaxMoves)
	}
	if b.Derived.Wins[0] && b.Derived.Wins[1] {
		return "the Queens from both players were surrounded at the same time"
	}
	return "unknown reason!?"
}

func (b *Board) Draw() bool {
	return b.IsFinished() && b.Derived.Wins[0] == b.Derived.Wins[1]
}

// Winner returns the player that wins on the current board.
// If it is a Draw or the match is not finished, return PlayerInvalid.
func (b *Board) Winner() PlayerNum {
	if !b.IsFinished() || b.Draw() {
		return PlayerInvalid
	}
	if b.Derived.Wins[0] {
		return PlayerFirst
	} else {
		return PlayerSecond
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

func (b *Board) EnumeratePieces(cb func(player PlayerNum, piece PieceType, pos Pos, covered bool)) {
	for pos, stack := range b.board {
		covered := false
		var player PlayerNum
		var piece PieceType
		for stack != 0 {
			stack, player, piece = stack.PopPiece()
			cb(player, piece, pos, covered)
			covered = true
		}
	}
}
