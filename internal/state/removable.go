package state

import (
	"github.com/janpfeifer/hiveGo/internal/generics"
)

// findArticulationPointsState stores the information of the graph used by the method FindArticulationPoints.
//
// Except for findArticulationPointsState.positions, everything else is seen in terms of a graph.
type findArticulationPointsState struct {
	numVertices    uint8
	isArticulation []bool
	allEdgesTarget []uint8
	edgesPerNode   [][2]uint8 // Shaped [node, 2], it holds the start and end indices into allEdgesTarget for each node.
	tIn, tLow      []uint8
}

// RemovablePositions returns set of removablePositions, without breaking the hive, pieces.
//
// Notice the result of this is stored in Derived.RemovablePositions, and it shouldn't be called directly, except
// for testing or benchmarking.
//
// It uses the popular linear algorithm to find the articulation points in a graph.
//
// Root is optional, and if given, it will start the DFS from the root.
func (b *Board) RemovablePositions(rootPos ...Pos) generics.Set[Pos] {
	if len(b.board) <= 1 {
		return nil
	}
	ap := &findArticulationPointsState{
		numVertices:    uint8(b.NumPiecesOnBoard()),
		allEdgesTarget: make([]uint8, 0, NumNeighbors*b.NumPiecesOnBoard()),
		edgesPerNode:   make([][2]uint8, b.NumPiecesOnBoard()),
	}

	// Enumerate positions and create a reverse map.
	positions := generics.KeysSlice(b.board)
	posToNodeIdx := make(map[Pos]uint8, len(positions))
	for nodeIdx, pos := range positions {
		//fmt.Printf("#%d: pos=%s\n", nodeIdx, pos)
		posToNodeIdx[pos] = uint8(nodeIdx)
	}
	var root uint8 // Default to 0
	if len(rootPos) > 0 {
		if nodeIdx, found := posToNodeIdx[rootPos[0]]; found {
			root = nodeIdx
		}
	}

	// Build edges.
	for nodeIdx, pos := range positions {
		ap.edgesPerNode[nodeIdx][0] = uint8(len(ap.allEdgesTarget))
		for neighbour := range b.OccupiedNeighboursIter(pos) {
			if toNodeIdx, found := posToNodeIdx[neighbour]; found {
				ap.allEdgesTarget = append(ap.allEdgesTarget, uint8(toNodeIdx))
			}
		}
		ap.edgesPerNode[nodeIdx][1] = uint8(len(ap.allEdgesTarget))
	}

	// Find articulation nodes.
	ap.FindArticulationPoints(root)

	// Enumerate non-articulate points as removable.
	removable := generics.MakeSet[Pos](int(b.NumPiecesOnBoard()))
	for nodeIdx, isAritculation := range ap.isArticulation {
		if !isAritculation {
			removable.Insert(positions[nodeIdx])
		}
	}
	return removable
}

// FindArticulationPoints O(N+M), N = #nodes, M = #edges,
// see description in https://cp-algorithms.com/graph/cutpoints.html
//
// It works by doing DFS and monitoring the "time of entry into node", let's call it t, and it is incremented
// as each node is visited.
//
// For each node we keep track of tIn -> time that node was visited, and tLow the time of the node with the lowest
// looping connection -- or itself, if it hasn't found a loop-back.
func (ap *findArticulationPointsState) FindArticulationPoints(root uint8) {
	if ap.numVertices == 0 {
		return
	}
	if ap.numVertices == 1 {
		ap.isArticulation[0] = true
		return
	}

	// state of the algorithm:
	ap.tIn = make([]uint8, ap.numVertices)
	ap.tLow = make([]uint8, ap.numVertices)
	ap.isArticulation = make([]bool, ap.numVertices)

	// DFS starting from root: the root visit is different than the rest, so we do it here.
	t := uint8(1)
	ap.tIn[root] = 1
	ap.tLow[root] = 1
	t++
	dfsChildren := 0
	for _, neighbour := range ap.allEdgesTarget[ap.edgesPerNode[root][0]:ap.edgesPerNode[root][1]] {
		if ap.tIn[neighbour] != 0 {
			continue
		}
		dfsChildren++
		t = ap.dfsVisit(root, neighbour, t)
	}
	// The root is an articulation point if it had to traverse through more than one neighbour on the DFS.
	// If it were not an articulation point, all nodes would have been reached from the first descendant.
	ap.isArticulation[root] = dfsChildren > 1
}

// dfsVisit returns the update time t.
func (ap *findArticulationPointsState) dfsVisit(from, to, t uint8) uint8 {
	ap.tIn[to] = t
	ap.tLow[to] = t
	ap.isArticulation[to] = false
	t++
	for _, neighbour := range ap.allEdgesTarget[ap.edgesPerNode[to][0]:ap.edgesPerNode[to][1]] {
		if neighbour == from {
			continue
		}
		if ap.tIn[neighbour] != 0 {
			// "Back-edge", an edge to node already visited: we take this into account to our tLow.
			ap.tLow[to] = min(ap.tLow[to], ap.tIn[neighbour])
			continue
		}
		t = ap.dfsVisit(to, neighbour, t)
		ap.tLow[to] = min(ap.tLow[to], ap.tLow[neighbour])
		if ap.tLow[neighbour] >= ap.tIn[to] {
			ap.isArticulation[to] = true
		}
	}
	//fmt.Printf("\t#%d: tIn=%d, tLow=%d, isArticulation=%v\n", to, ap.tIn[to], ap.tLow[to], ap.isArticulation[to])
	return t
}
