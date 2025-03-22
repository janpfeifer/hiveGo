package state

import (
	"encoding/gob"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"github.com/pkg/errors"
	"iter"
	"k8s.io/klog/v2"
	"maps"
	"sort"
)

var _ = fmt.Print

// PieceType currently limited to the 5 basic types plus a NoPiece, the null value.
type PieceType uint8

const (
	NoPiece PieceType = iota
	ANT
	BEETLE
	GRASSHOPPER
	QUEEN
	SPIDER
	LastPiece
)

const (
	// NumPlayers currently limited to 2.
	NumPlayers = 2

	// NumNeighbors of each position: the board is hexagonal.
	NumNeighbors = 6

	// NumPieceTypes doesn't include the NoPiece type.
	NumPieceTypes = LastPiece - 1

	// DefaultMaxMoves after which the game is considered a draw.
	DefaultMaxMoves = 100
)

// PlayerNum is the either 0 or 1 corresponding to the first player to move or the second player to move.
type PlayerNum uint8

const (
	PlayerFirst PlayerNum = iota
	PlayerSecond

	// PlayerInvalid represents an invalid PlayerNum.
	PlayerInvalid
)

//go:generate go tool enumer -type=PlayerNum -trimprefix=Player -values -text -json -yaml state.go

var (
	PieceLetters  = [LastPiece]string{"-", "A", "B", "G", "Q", "S"}
	LetterToPiece = map[string]PieceType{"A": ANT, "B": BEETLE, "G": GRASSHOPPER, "Q": QUEEN, "S": SPIDER}
	PieceNames    = [LastPiece]string{
		"None", "Ant", "Beetle", "Grasshopper", "Queen", "Spider",
	}

	// Pieces enumerates all the pieces, skipping the "NoPiece".
	Pieces = [NumPieceTypes]PieceType{ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER}
)

// String returns the long piece name.
func (p PieceType) String() string {
	return PieceNames[p]
}

// Availability represents the number of each piece type available to a player to put on the board.
type Availability [5]uint8

// InitialAvailability at the start of a match. See also TotalPiecesPerPlayer.
var InitialAvailability = Availability{3, 2, 3, 1, 2}

// TotalPiecesPerPlayer is the sum of the InitialAvailability.
const TotalPiecesPerPlayer = 11

// Pos packages x, y position.
type Pos [2]int8

// AbsInt8 returns the absolute value of an int8.
func AbsInt8(x int8) int8 {
	y := x >> 7
	return (x ^ y) - y
}

// X coordinate of the position.
func (pos Pos) X() int8 {
	return pos[0]
}

// Y coordinate of the position.
func (pos Pos) Y() int8 {
	return pos[1]
}

// Distance returns the manhattan distance of two positions.
func (pos Pos) Distance(pos2 Pos) int {
	return int(AbsInt8(pos[0]-pos2[0])) + int(AbsInt8(pos[1]-pos2[1]))
}

// Equal returns whether positions are the same.
func (pos Pos) Equal(pos2 Pos) bool {
	return pos == pos2
}

// String returns a text representation of Pos.
func (pos Pos) String() string {
	return fmt.Sprintf("(%d, %d)", pos[0], pos[1])
}

// FromDisplayPos converts a "display coordinate" position to the state coordinate.
// This is used by UI libraries, as the "display coordinate" is friendlier for humans.
func (pos Pos) FromDisplayPos() Pos {
	deltaY := pos[0] >> 1
	return Pos{pos[0], pos[1] - deltaY}
}

// ToDisplayPos converts a state coordinate to a "display coordinate".
// This is used by UI libraries, as the "display coordinate" is friendlier for humans.
func (pos Pos) ToDisplayPos() Pos {
	deltaY := pos[0] >> 1
	return Pos{pos[0], pos[1] + deltaY}
}

// SortPositions sorts according to y first and then x.
func SortPositions(positions []Pos) {
	sort.Slice(positions, func(i, j int) bool {
		if positions[i][1] != positions[j][1] {
			return positions[i][1] < positions[j][1]
		} else {
			return positions[i][0] < positions[j][0]
		}
	})
}

// PosSlice is a slice f Pos.
type PosSlice []Pos

func (p PosSlice) toDisplayPos() {
	for ii := range p {
		p[ii] = p[ii].ToDisplayPos()
	}
}
func (p PosSlice) fromDisplayPos() {
	for ii := range p {
		p[ii] = p[ii].FromDisplayPos()
	}
}

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
// If there is no pieces at given position, it returns NoPiece.
func (stack EncodedStack) PieceAt(stackPos uint8) (player PlayerNum, piece PieceType) {
	shift := stackPos << 3
	piece = PieceType((stack >> shift) & 0x7F)
	player = PlayerNum((stack >> (shift + 7)) & 1)
	return
}

// Top returns the player, piece at top.
func (stack EncodedStack) Top() (player PlayerNum, piece PieceType) {
	return stack.PieceAt(0)
}

// HasPiece returns whether there is a piece in the given stack of pieces.
func (stack EncodedStack) HasPiece() bool {
	return (stack & 0x7F) != 0
}

// HasQueen returns whether there is a queen in the given stack of pieces and return the player
// owning it.
func (stack EncodedStack) HasQueen() (bool, PlayerNum) {
	for stack != 0 {
		if PieceType(stack&0x7F) == QUEEN {
			return true, PlayerNum(stack >> 7 & 1)
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
func (stack EncodedStack) StackPiece(player PlayerNum, piece PieceType) EncodedStack {
	// TODO: check if stack is full, and if player can only be 0 or 1.
	return (stack << 8) | EncodedStack(piece&0x7F) | EncodedStack((player&1)<<7)
}

// PopPiece removes piece from top of the stak and returns the updated stack
// and the player/piece popped.
func (stack EncodedStack) PopPiece() (newStack EncodedStack, player PlayerNum, piece PieceType) {
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

// Board is a compact representation of the game state. It's compact to allow fast/cheap
// search on the space, by creating clones of it.
// Use it through methods that decode the packaged data.
type Board struct {
	available            [NumPlayers]Availability
	board                map[Pos]EncodedStack
	MoveNumber, MaxMoves int
	NextPlayer           PlayerNum

	// Derived information is regenerated after each move.
	Derived *Derived
}

// NewBoard creates a new empty board, with the correct initial number of pieces.
func NewBoard() *Board {
	board := &Board{
		available: [NumPlayers]Availability{
			InitialAvailability, InitialAvailability},
		board:      map[Pos]EncodedStack{},
		MoveNumber: 1,
		MaxMoves:   DefaultMaxMoves,
		NextPlayer: 0,
	}
	board.BuildDerived()
	return board
}

// Clone makes a deep copy of the board for a next move. The new Board.Previous
// is set to the current one, b.
func (b *Board) Clone() *Board {
	newB := &Board{}
	*newB = *b
	newB.Derived = nil
	newB.board = maps.Clone(b.board)
	return newB
}

// OpponentPlayer returns the player that is not the next one to play.
func (b *Board) OpponentPlayer() PlayerNum {
	return 1 - b.NextPlayer
}

// Available returns how many pieces of the given type are available for the given player.
func (b *Board) Available(player PlayerNum, piece PieceType) uint8 {
	return b.available[player][piece-1]
}

// SetAvailable sets the number of pieces available for the given type for the
// given player.
func (b *Board) SetAvailable(player PlayerNum, piece PieceType, value uint8) {
	b.available[player][piece-1] = value
}

// HasPiece returns whether there is a piece on the given location of the board.
func (b *Board) HasPiece(pos Pos) bool {
	stack, ok := b.board[pos]
	return ok && stack.HasPiece()
}

// PieceAt returns the piece at the top of the stack on the given position.
func (b *Board) PieceAt(pos Pos) (player PlayerNum, piece PieceType, stacked bool) {
	stack, _ := b.board[pos]
	player, piece = stack.PieceAt(0)
	stacked = (stack & 0x7F00) != 0
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
func (b *Board) StackPiece(pos Pos, player PlayerNum, piece PieceType) {
	stack, _ := b.board[pos]
	stack = stack.StackPiece(player, piece)
	b.board[pos] = stack
}

// PopPiece pops the piece at the given location, and returns it.
func (b *Board) PopPiece(pos Pos) (player PlayerNum, piece PieceType) {
	var stack EncodedStack
	stack, player, piece = b.board[pos].PopPiece()
	if stack != 0 {
		b.board[pos] = stack
	} else {
		delete(b.board, pos)
	}
	return
}

// NumPiecesOnBoard is the number of currently placed pieces.
func (b *Board) NumPiecesOnBoard() int8 {
	return int8(len(b.board))
}

// UsedLimits returns the max/min of x/y used in the board. Stores copy
// in Derived, to be reused if needed.
func (b *Board) UsedLimits() (minX, maxX, minY, maxY int8) {
	if b.Derived != nil {
		return b.Derived.MinX, b.Derived.MaxX, b.Derived.MinY, b.Derived.MaxY
	}
	first := true
	for pos := range b.board {
		x, y := pos.X(), pos.Y()
		if first || x > maxX {
			maxX = x
		}
		if first || x < minX {
			minX = x
		}
		if first || y > maxY {
			maxY = y
		}
		if first || y < minY {
			minY = y
		}
		first = false
	}
	return
}

// DisplayUsedLimits returns the max/min of x/y of the display in "display coordinates".
func (b *Board) DisplayUsedLimits() (minX, maxX, minY, maxY int8) {
	first := true
	for pos := range b.board {
		displayPos := pos.ToDisplayPos()
		x, y := displayPos.X(), displayPos.Y()
		if first || x > maxX {
			maxX = x
		}
		if first || x < minX {
			minX = x
		}
		if first || y > maxY {
			maxY = y
		}
		if first || y < minY {
			minY = y
		}
		first = false
	}
	return
}

// OccupiedPositions returns all the positions used.
func (b *Board) OccupiedPositions() []Pos {
	return generics.KeysSlice(b.board)
}

// OccupiedPositionsIter iterates over all positions used.
func (b *Board) OccupiedPositionsIter() iter.Seq[Pos] {
	return maps.Keys(b.board)
}

var neighborRelPositions = [6]Pos{{0, -1}, {1, -1}, {1, 0}, {0, 1}, {-1, 1}, {-1, 0}}

// NeighboursIter iterates over the 6 neighbor positions of the reference position.
//
// The iteration is properly ordered to match the direction.
//
// Also, the neighbors are listed in a clockwise manner.
func (pos Pos) NeighboursIter() iter.Seq[Pos] {
	return func(yield func(Pos) bool) {
		x, y := pos[0], pos[1]
		for _, relPos := range neighborRelPositions {
			if !yield(Pos{x + relPos[0], y + relPos[1]}) {
				return
			}
		}
	}
}

// Neighbours returns the 6 neighbour positions of the reference position. It
// returns a newly allocated slice.
//
// The list is properly ordered to match the direction. So if one takes Neighbours()[2] multiple
// times, one would move in straight line in the map.
//
// The neighbours are listed in a clockwise manner.
func (pos Pos) Neighbours() []Pos {
	x, y := pos[0], pos[1]
	return []Pos{
		{x, y - 1}, {x + 1, y - 1}, {x + 1, y},
		{x, y + 1}, {x - 1, y + 1}, {x - 1, y}}

	// Version with on display X,Y axis.
	// if x%2 == 0 {
	// 	return []Pos{
	// 		{x, y - 1}, {x + 1, y - 1}, {x + 1, y},
	// 		{x, y + 1}, {x - 1, y}, {x - 1, y - 1}}
	// } else {
	// 	return []Pos{
	// 		{x, y - 1}, {x + 1, y}, {x + 1, y + 1},
	// 		{x, y + 1}, {x - 1, y + 1}, {x - 1, y}}
	// }
}

// genericsIterFilter returns an iterators that only iterates over the values for which
// the filterFn returns true.
func genericsIterFilter[V any](seq iter.Seq[V], filterFn func(v V) bool) iter.Seq[V] {
	return func(yield func(V) bool) {
		for v := range seq {
			if filterFn(v) {
				if !yield(v) {
					return
				}
			}
		}
	}
}

// FilterPositionSlices filters the given positions according to the given filter.
// It destroys the contents of the provided slice and reuses the allocated space
// for the returned slice.
//
// FilterPositionSlices will preserve the elements that `filter(pos)` returns true,
// and discard the ones it returns false.
func FilterPositionSlices(positions []Pos, filter func(pos Pos) bool) (filtered []Pos) {
	filtered = positions[:0]
	for _, pos := range positions {
		if filter(pos) {
			filtered = append(filtered, pos)
		}
	}
	return
}

// OccupiedNeighbours returns the slice of positions with occupied neighbours.
func (b *Board) OccupiedNeighbours(pos Pos) (positions []Pos) {
	positions = pos.Neighbours()
	positions = FilterPositionSlices(positions, func(p Pos) bool { return b.HasPiece(p) })
	return
}

// OccupiedNeighboursIter iterate over the occupied neighbours.
func (b *Board) OccupiedNeighboursIter(pos Pos) iter.Seq[Pos] {
	return genericsIterFilter(pos.NeighboursIter(), func(p Pos) bool { return b.HasPiece(p) })
}

// EmptyNeighbours will return the slice of positions with empty neighbours.
func (b *Board) EmptyNeighbours(pos Pos) (positions []Pos) {
	positions = pos.Neighbours()
	positions = FilterPositionSlices(positions, func(p Pos) bool { return !b.HasPiece(p) })
	return
}

// EmptyNeighboursIter iterates over the empty neighbours.
func (b *Board) EmptyNeighboursIter(pos Pos) iter.Seq[Pos] {
	return genericsIterFilter(pos.NeighboursIter(), func(p Pos) bool { return !b.HasPiece(p) })
}

func (b *Board) PlayerNeighbours(player PlayerNum, pos Pos) (positions []Pos) {
	positions = pos.Neighbours()
	positions = FilterPositionSlices(positions, func(p Pos) bool {
		posPlayer, piece, _ := b.PieceAt(p)
		return piece != NoPiece && player == posPlayer
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
func (b *Board) OpponentNeighbours(pos Pos) (positions []Pos) {
	return b.PlayerNeighbours(b.OpponentPlayer(), pos)
}

// Save versions: over time I keep adding new fields.
const (
	ActionsScores = iota
	ActionsScoresAndActionsLabels
)

// LoadMatch restores match initial board, actions and scores.
func LoadMatch(dec *gob.Decoder) (initial *Board, actions []Action, scores []float32, actionsLabels [][]float32, err error) {
	initial = NewBoard()
	err = dec.Decode(&initial.MaxMoves)
	if err != nil {
		return
	}
	var saveFileVersion int
	if initial.MaxMoves > 0 {
		// First version, before the file version was saved along.
		saveFileVersion = ActionsScores
	} else {
		// MaxMoves <= 0 is the trigger that says we are saving
		// the saveFileVersion.
		if err = dec.Decode(&saveFileVersion); err != nil {
			return
		}
		klog.V(2).Infof("Loading saveFileVersion %d", saveFileVersion)
		if err = dec.Decode(&initial.MaxMoves); err != nil {
			return
		}
	}

	// Retrieve actions: all save file versions have it.
	actions = make([]Action, 0, initial.MaxMoves)
	if err = dec.Decode(&actions); err != nil {
		return
	}

	// Retrieve scores: one per board, so len(actions)+1
	scores = make([]float32, 0, initial.MaxMoves)
	if err = dec.Decode(&scores); err != nil {
		return
	}

	// ActionsLabels, for newer versions.
	if saveFileVersion == ActionsScoresAndActionsLabels {
		if err = dec.Decode(&actionsLabels); err != nil {
			return
		}
	}
	klog.V(2).Infof("Loaded MaxMoves=%d, %d scores, %d actions, %d actionsLabels",
		initial.MaxMoves, len(scores), len(actions), len(actionsLabels))
	return
}

// Encoder is any type of encoder -- implemented by gob.Encoder, json.Encoder
type Encoder interface {
	// Encode v or return an error.
	Encode(v any) error
}

// EncodeMatch will "save" (encode) the match and scores for future reconstruction.
// scores is optional.
func EncodeMatch(enc Encoder, MaxMoves int, actions []Action, scores []float32, actionsLabels [][]float32) error {
	saveFileVersion := -1
	if err := enc.Encode(saveFileVersion); err != nil {
		return fmt.Errorf("failed to encode match's board: %v", err)
	}
	saveFileVersion = ActionsScoresAndActionsLabels
	if err := enc.Encode(saveFileVersion); err != nil {
		return errors.Wrapf(err, "failed to encode match's board")
	}
	if err := enc.Encode(MaxMoves); err != nil {
		return errors.Wrapf(err, "failed to encode match's MaxMoves")
	}
	if err := enc.Encode(actions); err != nil {
		return errors.Wrapf(err, "failed to encode match's actions")
	}
	if err := enc.Encode(scores); err != nil {
		return errors.Wrapf(err, "failed to encode match's scores")
	}
	if err := enc.Encode(actionsLabels); err != nil {
		return errors.Wrapf(err, "failed to encode match's actionsLabels")
	}
	return nil
}
