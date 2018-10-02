package state

import (
	"log"
	"strings"

	"github.com/golang/glog"
)

// removable returns set of removable, without breaking the hive, pieces.
//
// The algorithm first finds out which pieces are in a loop or not, and
// then makes a final decision on whether they are removable.
func (b *Board) removable() (removable map[Pos]bool) {
	return convertToRemovables(b, updateLoopInfo(b))
}

func convertToRemovables(b *Board, loopInfo map[Pos]*rmNode) (removable map[Pos]bool) {
	removable = make(map[Pos]bool, b.NumPiecesOnBoard())
	for pos, node := range loopInfo {
		_, _, stacked := b.PieceAt(pos)
		if stacked || node.IsRemovable() {
			removable[pos] = true
		}
	}
	return
}

// Used for testing removable.
func (b *Board) TestRemovable(initialPos Pos) (removable map[Pos]bool) {
	return convertToRemovables(b, updateLoopInfoWithPos(b, initialPos))
}

// Used for testing older version of removable.
func (b *Board) TestOldRemovable() (removable map[Pos]bool) {
	removable = make(map[Pos]bool)
	for pos, _ := range b.board {
		if b.IsRemovable(pos) {
			removable[pos] = true
		}
	}
	return
}

// rmNode is the removable information: it informs about which neighbours
// are connected.
type rmNode struct {
	// Neighbours, in order. The first one will be the parent in DFS search.
	N []Pos

	// Connections. One per neighbour in N. If C[i] != i it means that there
	// is a connecting loop using neigbours N[i] and N[C[i]]. If a neighbour
	// is connected to more than one N, it points to the lowest i (including
	// itself).
	C []int8
}

// newRmNode creates a new rmNode with the neighbours properly
// set up. And if from is one of the neighbours, it will be in
// position 0.
func newRmNode(b *Board, pos, from Pos) *rmNode {
	node := &rmNode{
		N: b.OccupiedNeighbours(pos),
	}
	i := node.FindN(from)
	if i > 0 {
		node.N[0], node.N[i] = node.N[i], node.N[0]
	}
	node.C = make([]int8, len(node.N))
	for iC := range node.C {
		// Initially each neighbour is only connected to itself.
		node.C[iC] = int8(iC)
	}
	return node
}

func (node *rmNode) FindN(n Pos) int8 {
	for ii, n2 := range node.N {
		if n2 == n {
			return int8(ii)
		}
	}
	return -1
}

func (node *rmNode) IsRemovable() bool {
	for _, c := range node.C {
		if c != 0 {
			return false
		}
	}
	return true
}

func (node *rmNode) ConnectN(n1, n2 Pos) {
	i1 := node.FindN(n1)
	if i1 < 0 {
		log.Panicf("Position %s is not a neighbour.", n1)
	}
	i2 := node.FindN(n2)
	if i2 < 0 {
		log.Panicf("Position %s is not a neighbour.", n2)
	}
	node.ConnectIdx(i1, i2)
}

func (node *rmNode) ConnectIdx(i1, i2 int8) {
	if node.C[i1] == node.C[i2] {
		// They are already connected.
		return
	}

	// Find lowest target.
	lowest := i1
	if i2 < lowest {
		lowest = i2
	}
	iC1 := node.C[i1]
	iC2 := node.C[i2]
	if iC1 < lowest {
		lowest = iC1
	}
	if iC2 < lowest {
		lowest = iC2
	}

	// Convert all those connected to i1 and i2 to the lowest.
	for i, iC := range node.C {
		if iC == iC1 || iC == iC2 {
			node.C[i] = lowest
		}
	}
}

type recursiveInfo struct {
	b        *Board
	stack    []Pos
	stackMap map[Pos]int

	// stackConnect has a pointer to the higher position
	// in the stack at which the positions nieghbours have
	// been connected already. It's an optimization
	// to accelerate climbing the stack in cases loops
	// have already been registered.
	stackConnect []int

	loopInfo map[Pos]*rmNode
}

// Push position to stack.
func (ri *recursiveInfo) pushToStack(pos Pos) {
	ri.stackMap[pos] = len(ri.stack)
	ri.stackConnect = append(ri.stackConnect, len(ri.stack))
	ri.stack = append(ri.stack, pos)
}

// Pop position from stack.
func (ri *recursiveInfo) popFromStack() {
	pos := ri.stack[len(ri.stack)-1]
	ri.stack = ri.stack[0 : len(ri.stack)-1]
	delete(ri.stackMap, pos)
	ri.stackConnect = ri.stackConnect[0 : len(ri.stackConnect)-1]

	// Since the top of the stack will look into a new neighbour, it
	// should indicate that it hasn't been connected yet.
	if len(ri.stackConnect) > 0 {
		ri.stackConnect[len(ri.stackConnect)-1] = len(ri.stackConnect) - 1
	}
}

// updateLoopInfo performs a DFS, updating when it finds loops.
func updateLoopInfo(b *Board) map[Pos]*rmNode {
	// Pick any starting position.
	var pos Pos
	for pos = range b.board {
		break
	}
	return updateLoopInfoWithPos(b, pos)
}

func updateLoopInfoWithPos(b *Board, pos Pos) map[Pos]*rmNode {
	ri := &recursiveInfo{
		b:            b,
		stack:        make([]Pos, 0, b.NumPiecesOnBoard()),
		stackMap:     make(map[Pos]int, b.NumPiecesOnBoard()),
		stackConnect: make([]int, 0, b.NumPiecesOnBoard()),
		loopInfo:     make(map[Pos]*rmNode, b.NumPiecesOnBoard()),
	}

	recursivelyUpdateLoopInfo(ri, pos, pos)
	return ri.loopInfo
}

func recursivelyUpdateLoopInfo(ri *recursiveInfo, pos, from Pos) {
	glog.V(2).Infof("Visiting %s (from %s)", pos, from)
	node := newRmNode(ri.b, pos, from)
	ri.loopInfo[pos] = node
	ri.pushToStack(pos)
	first := 1 // Skip the "from" node.
	if from == pos {
		first = 0
	}
	for ii := first; ii < len(node.N); ii++ {
		neighbour := node.N[ii]
		if loopNode, visited := ri.loopInfo[neighbour]; visited {
			stackStart, isInStack := ri.stackMap[neighbour]
			if !isInStack {
				// If neighbour is not in stack, it has already been visited and
				// accounted for (because it was reached by the other end)
				glog.V(2).Infof("  Skipping visit to %s, already visited", neighbour)
				continue
			}

			// Connect node for pos back to ancestor in stack.
			node.ConnectIdx(0, int8(ii))

			// Connect everyone in path in stack.
			// TODO: keep tabs of connections already made and jump them.
			for stackIdx := len(ri.stack) - 2; stackIdx > stackStart; stackIdx-- {
				if ri.stackConnect[stackIdx] < stackIdx {
					// Skip directly to position already connected.
					stackIdx = ri.stackConnect[stackIdx] + 1
					if stackStart < ri.stackConnect[stackIdx] {
						// Register the new stack position up to which all will be connected.
						ri.stackConnect[stackIdx] = stackStart
					}
					continue
				}

				// Connects neighbour coming from the top of the stak to the neighbour coming
				// from the bottom of the stack.
				stackPos := ri.stack[stackIdx]
				stackNode := ri.loopInfo[stackPos]
				stackNode.ConnectN(ri.stack[stackIdx-1], ri.stack[stackIdx+1])

				// Register the stack position up to which all will be connected.
				ri.stackConnect[stackIdx] = stackStart
			}

			// Looping point in stack.
			loopNode.ConnectN(ri.stack[stackStart+1], pos)

			// Debug.
			if glog.V(2) {
				var loop []Pos
				loop = append(loop, pos)
				for stackIdx := len(ri.stack) - 2; stackIdx > stackStart; stackIdx-- {
					stackPos := ri.stack[stackIdx]
					loop = append(loop, stackPos)
				}
				loop = append(loop, neighbour)
				glog.V(2).Infof("  Loop found: %s", strings.Join(PosStrings(loop), ", "))
			}

		} else {
			recursivelyUpdateLoopInfo(ri, neighbour, pos)
		}
	}

	glog.V(2).Infof("  Popping back to %s", from)
	ri.popFromStack()
}
