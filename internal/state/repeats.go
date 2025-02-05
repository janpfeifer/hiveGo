// This file contains the functions that check if a match position
// is repeated, and a cache of previous states of matches, to accelerate
// search in some cases.
package state

import (
	"encoding/binary"
	"hash/fnv"
	"log"
	"sort"
)

// PosStack represents a position and the stack of pieces in the position.
type PosStack struct {
	Pos   Pos
	Stack EncodedStack
}

// PosStackSlice is a sortable slice of PosStack.
type PosStackSlice []PosStack

func (p PosStackSlice) Len() int { return len(p) }
func (p PosStackSlice) Less(i, j int) bool {
	if p[i].Pos[1] != p[j].Pos[1] {
		return p[i].Pos[1] < p[j].Pos[1]
	} else {
		return p[i].Pos[0] < p[j].Pos[0]
	}
}
func (p PosStackSlice) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

// normalizedPosStackSlice returns sorted slice of positions/stacks that
// has been shifted to start at (0,0) or (1,0), if it starts at an odd
// X position.
//
// Return value should be stored in Board.Derived, and should be accessed
// from there.
func (b *Board) normalizedPosStackSlice() (poss PosStackSlice) {
	poss = make(PosStackSlice, 0, len(b.board))
	for pos, stack := range b.board {
		poss = append(poss, PosStack{pos, stack})
	}

	// Normalize positions such that they start at (0,0) -- or (1,0), if it
	// starts at an odd position.
	minX, _, minY, _ := b.UsedLimits()
	if minX&1 != 0 {
		minX--
	}
	for ii := range poss {
		poss[ii].Pos[0] -= minX
		poss[ii].Pos[1] -= minY
	}

	// Finally sort positions.
	sort.Sort(poss)
	return poss
}

// normalizedHash will calculate a hash from a normalized board: pieces
// shifted to start at 0,0.
//
// It requires b.Derived to exist, and in particular
// b.Derived.NormalizedPosStackSlice to be filled.
func (b *Board) normalizedHash() uint64 {
	hasher := fnv.New64a()
	if len(b.Derived.NormalizedPosStackSlice) == 0 {
		return 0
	}
	if err := binary.Write(hasher, binary.LittleEndian, b.NextPlayer); err != nil {
		log.Panicf("Failed to write to hasher: %v", err)
	}
	if err := binary.Write(hasher, binary.LittleEndian, b.Derived.NormalizedPosStackSlice); err != nil {
		log.Panicf("Failed to write to hasher: %v", err)
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

	poss1 := b1.Derived.NormalizedPosStackSlice
	poss2 := b2.Derived.NormalizedPosStackSlice
	for ii := range poss1 {
		if poss1[ii] != poss2[ii] {
			return false
		}
	}
	return true
}

// FindRepeats returns the number of repeated board positions in the same match.
func (b *Board) FindRepeats() uint8 {
	numPieces := b.NumPiecesOnBoard()
	for b2 := b.Previous; b2 != nil && b2.NumPiecesOnBoard() == numPieces; b2 = b2.Previous {
		if CompareBoards(b, b2) {
			return b2.Derived.Repeats + 1
		}
	}
	return 0
}
