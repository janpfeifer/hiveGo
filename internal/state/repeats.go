package state

// This file contains the functions that check if a match position
// is repeated, and a cache of previous states of matches, to accelerate
// search in some cases.

import (
	"encoding/binary"
	"hash/fnv"
	"k8s.io/klog/v2"
	"slices"
)

// HashNode represents the a list (but during exploration it may become a tree) of hash
// of the previous matches in a line of the game, used to check for repeated positions.
type HashNode struct {
	Hash uint64
	Prev *HashNode
}

// PosStack represents a position and the stack of pieces in the position.
type PosStack struct {
	Pos   Pos
	Stack EncodedStack
}

// PosStackSlice is a sortable slice of PosStack.
type PosStackSlice []PosStack

func cmpInt8(a, b int8) int {
	if a < b {
		return -1
	}
	if b < a {
		return 1
	}
	return 0
}

// Sort in-place slice of PosStack.
// There should be only one stack per position.
func (s PosStackSlice) Sort() {
	slices.SortFunc(s, func(a, b PosStack) int {
		if cmp := cmpInt8(a.Pos[1], b.Pos[1]); cmp != 0 {
			return cmp
		}
		return cmpInt8(a.Pos[0], b.Pos[0])
	})
}

// normalizedPosStackSlice returns sorted slice of positions/stacks that
// has been shifted to start at (0,0) or (1,0), if it starts at an odd
// X position.
//
// Return value should be stored in Board.Derived, and should be accessed
// from there.
func (b *Board) normalizedPosStackSlice() (pieces PosStackSlice) {
	pieces = make(PosStackSlice, 0, len(b.board))
	for pos, stack := range b.board {
		pieces = append(pieces, PosStack{pos, stack})
	}

	// Normalize positions such that they start at (0,0) -- or (1,0), if it
	// starts at an odd position.
	minX, _, minY, _ := b.UsedLimits()
	if minX&1 != 0 {
		minX--
	}
	for ii := range pieces {
		pieces[ii].Pos[0] -= minX
		pieces[ii].Pos[1] -= minY
	}
	pieces.Sort()
	return pieces
}

// normalizedHash will calculate a hash from a normalized board: pieces
// shifted to start at 0,0, and iterated in X order than Y.
//
// It requires b.Derived to exist, and in particular
// b.Derived.NormalizedPosStackSlice to be filled.
func (b *Board) normalizedHash() uint64 {
	hasher := fnv.New64a()
	if len(b.Derived.NormalizedPosStackSlice) == 0 {
		return 0
	}
	if err := binary.Write(hasher, binary.LittleEndian, b.NextPlayer); err != nil {
		klog.Fatalf("Failed to write to hasher: %v", err)
	}
	if err := binary.Write(hasher, binary.LittleEndian, b.Derived.NormalizedPosStackSlice); err != nil {
		klog.Fatalf("Failed to write to hasher: %v", err)
	}
	return hasher.Sum64()
}

// CompareBoards checks if two boards are the same or equivalent --
// translation of board positions.
//
// This function assumed Board.Derived is set for both boards.
func CompareBoards(b1, b2 *Board) bool {
	if b1.Derived.Hash != b2.Derived.Hash {
		return false
	}
	if b1.NextPlayer != b2.NextPlayer || b1.NumPiecesOnBoard() != b2.NumPiecesOnBoard() {
		return false
	}
	minX1, maxX1, minY1, maxY1 := b1.UsedLimits()
	minX2, maxX2, minY2, maxY2 := b2.UsedLimits()
	if maxX1-minX1 != maxX2-minX2 || maxY1-minY1 != maxY2-minY2 {
		return false
	}

	stacks1 := b1.Derived.NormalizedPosStackSlice
	stacks2 := b2.Derived.NormalizedPosStackSlice
	for ii := range stacks1 {
		if stacks1[ii] != stacks2[ii] {
			return false
		}
	}
	return true
}

// CountRepeats returns the number of repeated board positions in the same match.
func (b *Board) CountRepeats() uint8 {
	h := b.Derived.Hash
	var repeats uint8
	for hn := b.PreviousBoards; hn != nil; hn = hn.Prev {
		if hn.Hash == h {
			repeats++
		}
	}
	return repeats
}
