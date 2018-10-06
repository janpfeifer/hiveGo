package state

import (
	"encoding/gob"
	"fmt"
	"sort"
)

var _ = fmt.Print

type Piece uint8

const (
	NO_PIECE Piece = iota
	ANT
	BEETLE
	GRASSHOPPER
	QUEEN
	SPIDER
	LAST_PIECE_TYPE
)

const (
	NUM_PLAYERS     = 2
	NUM_NEIGHBOURS  = 6
	NUM_PIECE_TYPES = LAST_PIECE_TYPE - 1 // Includes the "NO_PIECE" type.
)

var (
	PieceLetters  = [LAST_PIECE_TYPE]string{"-", "A", "B", "G", "Q", "S"}
	LetterToPiece = map[string]Piece{"A": ANT, "B": BEETLE, "G": GRASSHOPPER, "Q": QUEEN, "S": SPIDER}
	PieceNames    = [LAST_PIECE_TYPE]string{
		"None", "Ant", "Beetle", "Grasshopper", "Queen", "Spider",
	}

	// Pieces enumerates all the pieces, skipping the "NO_PIECE".
	Pieces = [NUM_PIECE_TYPES]Piece{ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER}
)

var INITIAL_AVAILABILITY = Availability{3, 2, 3, 1, 2}

const TOTAL_PIECES_PER_PLAYER = 11

// Pos packages x, y position.
type Pos [2]int8

// Array with counts of pieces for A, B, G, Q, S respectively
type Availability [5]uint8

// Board is a compact representation of the game state. It's compact to allow fast/cheap
// search on the space. Use it through methods that decode the packaged data.
type Board struct {
	available            [NUM_PLAYERS]Availability
	board                map[Pos]EncodedStack
	MoveNumber, MaxMoves int
	NextPlayer           uint8

	// Previous is a link to the Board at the previous position, or nil if
	// this is the initial Board.
	Previous *Board

	// Derived information is regenerated after each move.
	Derived *Derived
}

func (p Piece) String() string {
	return PieceNames[p]
}

func (pos Pos) X() int8 {
	return pos[0]
}

func (pos Pos) Y() int8 {
	return pos[1]
}

func (pos Pos) String() string {
	return fmt.Sprintf("(%d, %d)", pos[0], pos[1])
}

type PosSlice []Pos

func (p PosSlice) Len() int { return len(p) }
func (p PosSlice) Less(i, j int) bool {
	if p[i][1] != p[j][1] {
		return p[i][1] < p[j][1]
	} else {
		return p[i][0] < p[j][0]
	}
}
func (p PosSlice) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func PosSort(poss []Pos) {
	sort.Sort(PosSlice(poss))
}

func PosStrings(poss []Pos) []string {
	strs := make([]string, len(poss))
	for ii, pos := range poss {
		strs[ii] = fmt.Sprint(pos)
	}
	return strs
}

// EncodedStack represents a stacked collection of pieces, typically in some location of
// the map.
// See methods `HasPiece` and `PieceAt` for usage, and the implementation for encoding
// details and limitation (most 8 pieces can be stacked).
type EncodedStack uint64

// PieceAt returns player and piece at position pos in the stack.
// Pos 0 is the top of the stack, 1 is the first piece under it, etc.
// If there is no pieces at given position, it returns NO_PIECE.
func (stack EncodedStack) PieceAt(stackPos uint8) (player uint8, piece Piece) {
	shift := stackPos << 3
	player = uint8((stack >> (shift + 7)) & 1)
	piece = Piece((stack >> shift) & 0x7F)
	return
}

// Top returns the player, piece at top.
func (stack EncodedStack) Top() (player uint8, piece Piece) {
	return stack.PieceAt(0)
}

// HasPiece returns whether there is a piece in the given stack of pieces.
func (stack EncodedStack) HasPiece() bool {
	return (stack & 0x7F) != 0
}

// Returns whether there is a queen in the given stack of pieces and return the player
// owning it.
func (stack EncodedStack) HasQueen() (bool, uint8) {
	for stack != 0 {
		if Piece(stack&0x7F) == QUEEN {
			return true, uint8(stack >> 7 & 1)
		}
		stack >>= 8
	}
	return false, 0
}

// CountPieces returns the number of pieces stacked.
func (stack EncodedStack) CountPieces() (count uint8) {
	count = 0
	for stack != 0 {
		count++
		stack >>= 8
	}
	return
}

// StackPiece stacks piece and returns new stack value.
func (stack EncodedStack) StackPiece(player uint8, piece Piece) EncodedStack {
	// TODO: check if stack is full, and if player can only be 0 or 1.
	return (stack << 8) | EncodedStack(piece&0x7F) | EncodedStack((player&1)<<7)
}

// PopPiece removes piece from top of the stak and returns the udpated stack
// and the player/piece popped.
func (stack EncodedStack) PopPiece() (newStack EncodedStack, player uint8, piece Piece) {
	player, piece = stack.Top()
	newStack = stack >> 8
	return
}

// Count total number of pieces available.
func (a Availability) Count() (count uint8) {
	count = 0
	for _, value := range a {
		count += value
	}
	return
}

// NewBoard creates a new empty board, with the correct initial number of pieces.
func NewBoard() *Board {
	board := &Board{
		available: [NUM_PLAYERS]Availability{
			INITIAL_AVAILABILITY, INITIAL_AVAILABILITY},
		board:      map[Pos]EncodedStack{},
		MoveNumber: 1,
		MaxMoves:   1000,
		NextPlayer: 0,
		Previous:   nil,
	}
	board.BuildDerived()
	return board
}

// Copy makes a deep copy of the board for a next move. The new Board.Previous
// is set to the current one, b.
func (b *Board) Copy() *Board {
	newB := &Board{}
	*newB = *b
	newB.Derived = nil
	newB.Previous = b
	newB.board = make(map[Pos]EncodedStack)
	for pos, stack := range b.board {
		newB.board[pos] = stack
	}
	return newB
}

func (b *Board) OpponentPlayer() uint8 {
	return 1 - b.NextPlayer
}

// Available returns how many pieces of the given type are avaiable for the given player.
func (b *Board) Available(player uint8, piece Piece) uint8 {
	return b.available[player][piece-1]
}

// SetAvailable sets teh number of pieces available for the given type for the
// given player.
func (b *Board) SetAvailable(player uint8, piece Piece, value uint8) {
	b.available[player][piece-1] = uint8(value)
}

// HasPiece returns whether there is a piece on the given location of the board.
func (b *Board) HasPiece(pos Pos) bool {
	stack, ok := b.board[pos]
	return ok && stack.HasPiece()
}

func (b *Board) PieceAt(pos Pos) (player uint8, piece Piece, stacked bool) {
	stack, _ := b.board[pos]
	player, piece = stack.PieceAt(0)
	stacked = ((stack & 0x7F00) != 0)
	return
}

func (b *Board) CountAt(pos Pos) uint8 {
	if stack, ok := b.board[pos]; ok {
		return stack.CountPieces()
	}
	return 0
}

// StackAt returns the EncodedStack at given position at the board. It will return
// an empty stack if there is no position there.
func (b *Board) StackAt(pos Pos) (stack EncodedStack) {
	stack, _ = b.board[pos]
	return
}

// StackPiece adds a piece to the given board position. It doesn't subtract from
// the available list.
func (b *Board) StackPiece(pos Pos, player uint8, piece Piece) {
	stack, _ := b.board[pos]
	stack = stack.StackPiece(player, piece)
	b.board[pos] = stack
}

// PopPiece pops the piece at the given location, and returns it.
func (b *Board) PopPiece(pos Pos) (player uint8, piece Piece) {
	var stack EncodedStack
	stack, player, piece = b.board[pos].PopPiece()
	if stack != 0 {
		b.board[pos] = stack
	} else {
		delete(b.board, pos)
	}
	return
}

func (b *Board) NumPiecesOnBoard() int8 {
	return int8(len(b.board))
}

// UsedLimits returns the max/min of x/y used in the board. Stores copy
// in Derived, to be reused if needed.
func (b *Board) UsedLimits() (min_x, max_x, min_y, max_y int8) {
	if b.Derived != nil {
		return b.Derived.MinX, b.Derived.MaxX, b.Derived.MinY, b.Derived.MaxY
	}
	first := true
	for pos, _ := range b.board {
		x, y := pos.X(), pos.Y()
		if first || x > max_x {
			max_x = x
		}
		if first || x < min_x {
			min_x = x
		}
		if first || y > max_y {
			max_y = y
		}
		if first || y < min_y {
			min_y = y
		}
		first = false
	}
	return
}

// OccupiedPositions returns all the positions used.
func (b *Board) OccupiedPositions() (poss []Pos) {
	poss = make([]Pos, 0, len(b.board))
	for pos := range b.board {
		poss = append(poss, pos)
	}
	return poss
}

// Neighbours returns the 6 neighbour positions of the reference position. It
// returns a newly allocated slice.
//
// The list is properly ordered to match the direction. So if one takes Neighbours()[2] multiple
// times, one would move in straight line in the map.
//
// Also the neighbours are listed in a clockwise manner.
func (pos Pos) Neighbours() []Pos {
	x, y := pos[0], pos[1]
	if x%2 == 0 {
		return []Pos{
			Pos{x, y - 1}, Pos{x + 1, y - 1}, Pos{x + 1, y},
			Pos{x, y + 1}, Pos{x - 1, y}, Pos{x - 1, y - 1}}
	} else {
		return []Pos{
			Pos{x, y - 1}, Pos{x + 1, y}, Pos{x + 1, y + 1},
			Pos{x, y + 1}, Pos{x - 1, y + 1}, Pos{x - 1, y}}
	}
}

// FilterPositions filters the given positions according to the given filter.
// It destroys the contents of the provided slice and reuses the allocated space
// for the returned slice.
//
// FilterPositions will preserve the elements that `filter(pos)` returns true,
// and discard the ones it returns false.
func FilterPositions(positions []Pos, filter func(pos Pos) bool) (filtered []Pos) {
	filtered = positions[:0]
	for _, pos := range positions {
		if filter(pos) {
			filtered = append(filtered, pos)
		}
	}
	return
}

// OccupiedNeighbours will return the slice of positions with occupied neighbours.
func (b *Board) OccupiedNeighbours(pos Pos) (poss []Pos) {
	poss = pos.Neighbours()
	poss = FilterPositions(poss, func(p Pos) bool { return b.HasPiece(p) })
	return
}

// EmptyNeighbours will return the slice of positions with empty neighbours.
func (b *Board) EmptyNeighbours(pos Pos) (poss []Pos) {
	poss = pos.Neighbours()
	poss = FilterPositions(poss, func(p Pos) bool { return !b.HasPiece(p) })
	return
}

func (b *Board) PlayerNeighbours(player uint8, pos Pos) (poss []Pos) {
	poss = pos.Neighbours()
	poss = FilterPositions(poss, func(p Pos) bool {
		posPlayer, piece, _ := b.PieceAt(p)
		return piece != NO_PIECE && player == posPlayer
	})
	return
}

// FriendlyNeighbours will return the slice of neighbouring positions occupied by pieces
// of the b.NextPlayer.
func (b *Board) FriendlyNeighbours(pos Pos) (poss []Pos) {
	return b.PlayerNeighbours(b.NextPlayer, pos)
}

// OpponentNeighbours will return the slice of neighbouring positions occupied by opponents
// of the b.NextPlayer.
func (b *Board) OpponentNeighbours(pos Pos) (poss []Pos) {
	return b.PlayerNeighbours(b.OpponentPlayer(), pos)
}

// SaveMatch will "save" (encode) the match and scores for future reconstruction.
// scores is opional.
func SaveMatch(enc *gob.Encoder, initial *Board, actions []Action, scores []float32) error {
	b := initial.Copy()
	b.Derived = nil
	if err := enc.Encode(b); err != nil {
		return fmt.Errorf("Failed to encode match's board: %v", err)
	}
	if err := enc.Encode(actions); err != nil {
		return fmt.Errorf("Failed to encode match's actions: %v", err)
	}
	if err := enc.Encode(scores); err != nil {
		return fmt.Errorf("Failed to encode match's scores: %v", err)
	}
	return nil
}

// LoadMatch restores match initial board, actions and scores.
func LoadMatch(dec *gob.Decoder) (initial *Board, actions []Action, scores []float32, err error) {
	tmp := &Board{}
	err = dec.Decode(tmp)
	if err != nil {
		return
	}
	initial = NewBoard()
	initial.MaxMoves = tmp.MaxMoves
	err = dec.Decode(&actions)
	if err != nil {
		return
	}
	err = dec.Decode(&scores)
	return
}
