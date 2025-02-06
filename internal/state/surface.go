package state

// This file holds the definition of Surface of a hive. These are connected
// areas where an ANT can move. Usually there is only an external one,
// surrounding the hive. But there can more internal ones.

// Surfaces are connected areas of the hive space, where an ANT
// could presumably move.
type Surfaces struct {
	PosToSurface map[Pos]int // Maps position to surface it is part of.
	Num          int         // Number of surfaces.
}

var UnusedPos = Pos{127, 127}

// Surfaces create a new Surfaces object from Board position.
func (b *Board) Surfaces() (surfaces *Surfaces) {
	surfaces = &Surfaces{}

	for piecePos := range b.board {
		for _, pos := range b.EmptyNeighbours(piecePos) {
			if _, found := surfaces.PosToSurface[pos]; found {
				// Position already in surfaces.
				continue
			}

			// Perform a BFS to find all valid positions.
			toVisit := map[Pos]bool{pos: true}
			visited := map[Pos]bool{pos: true}
			for len(toVisit) > 0 {
				newToVisit := make(map[Pos]bool)
				for pos := range toVisit {
					for _, nextVisit := range b.EmptyAndConnectedNeighbours(pos, UnusedPos, visited) {
						visited[nextVisit] = true
						newToVisit[nextVisit] = true
					}
				}
				toVisit = newToVisit
			}

			// Assign visited positions to new surface.
			surfaceIndex := surfaces.Num
			surfaces.Num++
			for pos := range visited {
				surfaces.PosToSurface[pos] = surfaceIndex
			}
		}
	}
	return
}

func (b *Board) surfaceAntMoves(srcPos Pos, surfaces *Surfaces) (poss []Pos) {
	visited := map[Pos]bool{srcPos: true}
	visitedSurfaces := make(map[int]bool, surfaces.Num)
	for _, pos := range b.EmptyAndConnectedNeighbours(srcPos, srcPos, visited) {
		visitedSurfaces[surfaces.PosToSurface[pos]] = true
	}

	poss = make([]Pos, 0, len(surfaces.PosToSurface))
	for pos, surfaceNum := range surfaces.PosToSurface {
		if !visitedSurfaces[surfaceNum] {
			continue
		}
		// TODO: check position is not disconnected: it was only connected to moving
		//   ANT.
		if visited[pos] {

		}
	}
	return
}
