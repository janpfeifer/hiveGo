package state

import "fmt"

var _ = fmt.Printf

// EmptyAndConnectedNeighbours returns neighbouring positions that are empty but
// still connected to the graph of insects.
//
// It also checks that piece is not "squeezing" through two other pieces, or that
// it moves loosing touch to pieces -- that is, the intersection of neighboring
// pieces before and after the move must be non-empty.
//
// Args:
//
//	srcPos: from where this move starts.
//	originalPos: where the piece is going to leave from: this is equal to
//	  srcPos for the first step of a piece, and then something different
//	  later on. Presumably will be empty and therefore can't be considered as
//	  an occupied neighboor.
//	invalid: Set of positions not to consider, since they were already visited.
func (b *Board) EmptyAndConnectedNeighbours(srcPos, originalPos Pos, invalid map[Pos]bool) (poss []Pos) {
	poss = make([]Pos, 0, NumNeighbors)

	// Initialize neighbours and occupied predicate (assuming the piece will leave originalPos).
	neighbours := srcPos.NeighborsSlice()
	occupied := make([]bool, NumNeighbors)
	for ii := 0; ii < NumNeighbors; ii++ {
		occupied[ii] = b.HasPiece(neighbours[ii]) && neighbours[ii] != originalPos
	}

	// Find valid connections.
	for ii := 0; ii < NumNeighbors; ii++ {
		tgtPos := neighbours[ii]
		if invalid[tgtPos] {
			// Likely already visited.
			continue
		}
		if occupied[ii] {
			// Target destination must be empty.
			continue
		}
		positionLeftOfMoveOccupied := occupied[(ii+1)%NumNeighbors]
		positionRightOfMoveOccupied := occupied[(ii-1+NumNeighbors)%NumNeighbors]
		if positionLeftOfMoveOccupied && positionRightOfMoveOccupied {
			// Squeeze between two pieces is not allowed.
			continue
		}
		if !positionLeftOfMoveOccupied && !positionRightOfMoveOccupied {
			// But at least one of the two positions in between the source and target positions
			// must be occupied.
			continue
		}
		poss = append(poss, tgtPos)
	}
	return
}

// queenMoves enumerates the valid moves for the Queen located at the given position.
func (b *Board) queenMoves(srcPos Pos) (poss []Pos) {
	return b.EmptyAndConnectedNeighbours(srcPos, srcPos, make(map[Pos]bool))
}

// spiderMoves enumerates the valid moves for the Spider located at the given position.
func (b *Board) spiderMoves(srcPos Pos) (poss []Pos) {
	poss = nil
	endPos := map[Pos]bool{}
	visitedPath := map[Pos]bool{srcPos: true}
	b.spiderMovesDFS(srcPos, srcPos, 3, endPos, visitedPath)
	for pos := range endPos {
		poss = append(poss, pos)
	}
	return poss
}

// spiderMovesDFS traverse connected neighbours and keeps track of valid final destinations for
// a spider in endPos.
func (b *Board) spiderMovesDFS(srcPos, originalPos Pos, depth int, endPos, visitedPath map[Pos]bool) {
	depth--
	if depth == 0 {
		// When the number of moves are over, take the final steps.
		for _, pos := range b.EmptyAndConnectedNeighbours(srcPos, originalPos, visitedPath) {
			endPos[pos] = true
		}
	} else {
		// Recursively visited next steps.
		for _, pos := range b.EmptyAndConnectedNeighbours(srcPos, originalPos, visitedPath) {
			// Mark next step as visited and recurse.
			visitedPath[pos] = true
			b.spiderMovesDFS(pos, originalPos, depth, endPos, visitedPath)
			// Reset visited next step, because the same location can be reached by different steps
			// and it should be fine.
			delete(visitedPath, pos)
		}
	}
}

// grasshopperMoves enumerates the valid moves for the Grasshopper located at the given position.
func (b *Board) grasshopperMoves(srcPos Pos) (poss []Pos) {
	poss = nil
	for direction := 0; direction < NumNeighbors; direction++ {
		steps, tgtPos := b.grasshopperNextFree(srcPos, direction)
		if steps > 1 {
			poss = append(poss, tgtPos)
		}
	}
	return
}

func (b *Board) grasshopperNextFree(srcPos Pos, direction int) (steps int, tgtPos Pos) {
	steps = 0
	for tgtPos = srcPos; b.HasPiece(tgtPos); tgtPos = tgtPos.NeighborsSlice()[direction] {
		steps++
	}
	return
}

// antMoves enumerates the valid moves for the Ant located at the given position.
func (b *Board) antMoves(srcPos Pos) (poss []Pos) {
	// Perform a BFS to find all valid positions.
	toVisit := map[Pos]bool{srcPos: true}
	visited := map[Pos]bool{srcPos: true}
	for len(toVisit) > 0 {
		newToVisit := make(map[Pos]bool)
		for pos := range toVisit {
			for _, nextVisit := range b.EmptyAndConnectedNeighbours(pos, srcPos, visited) {
				visited[nextVisit] = true
				newToVisit[nextVisit] = true
			}
		}
		toVisit = newToVisit
	}

	// Collect all visited locations as valid moves, except the original one.
	poss = make([]Pos, 0, len(visited)-1)
	for pos := range visited {
		if pos != srcPos {
			poss = append(poss, pos)
		}
	}
	PosSort(poss)
	return
}

// beetleMoves enumerates the valid moves for the Beetle located at the given position.
func (b *Board) beetleMoves(srcPos Pos) (poss []Pos) {
	// If on top of a piece, it can move anywhere.
	if _, _, stacked := b.PieceAt(srcPos); stacked {
		return srcPos.NeighborsSlice()
	}

	// It can move onto any other piece.
	poss = b.OccupiedNeighbours(srcPos)

	// And it moves like the queen: notice that if not moving from the top,
	// it can't squeeze between pieces either.
	for _, pos := range b.EmptyAndConnectedNeighbours(srcPos, srcPos, nil) {
		poss = append(poss, pos)
	}
	return
}
