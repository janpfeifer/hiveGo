package ai_test

import (
	"fmt"
	"github.com/golang/glog"
	"log"
	"reflect"
	"testing"

	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

func printBoard(b *Board, action Action) {
	ui := ascii_ui.NewUI(true, false)
	ui.PrintBoard(b)
	fmt.Printf("Action: %s\n", action)
	if false {
		debugNeighboursForPos(ui, Pos{0, 0})
		debugNeighboursForPos(ui, Pos{1, 0})
	}
}

func isZero(f []float32) bool {
	for _, v := range f {
		if v != 0 {
			return false
		}
	}
	return true
}

// Assumes the center is in (0, 0)
func getPosition(pos, center Pos, f *ai.ActionPositionFeatures) []float32 {
	relPos := Pos{pos.X() - center.X(), pos.Y() - center.Y()}
	if relPos.X() == 0 && relPos.Y() == 0 {
		return f.Center
	}
	neighbourhood := &ai.X_EVEN_NEIGHBOURS
	if center.X()%2 != 0 {
		neighbourhood = &ai.X_ODD_NEIGHBOURS
	}

	for section := 0; section < 6; section++ {
		for ii := 0; ii < ai.POSITIONS_PER_SECTION; ii++ {
			neighPos := neighbourhood[section][ii]
			if relPos.Equal(neighPos) {
				return f.Sections[section][ii*ai.FEATURES_PER_POSITION : (ii+1)*ai.FEATURES_PER_POSITION]
			}
		}
	}
	log.Panicf("Position %s (rel: %s) not in neighbourhood of %s", pos, relPos, center)
	return nil
}

func TestPolicyFeatures(t *testing.T) {
	b := NewBoard()
	b.StackPiece(Pos{0, 0}, 0, QUEEN)
	b.StackPiece(Pos{0, 1}, 1, QUEEN)
	b.StackPiece(Pos{0, 1}, 1, BEETLE)
	b.StackPiece(Pos{0, 1}, 0, BEETLE)
	b.StackPiece(Pos{-1, -1}, 0, GRASSHOPPER)
	b.BuildDerived()
	action := Action{true, GRASSHOPPER, Pos{-1, -1}, Pos{1, 0}}
	printBoard(b, action)

	actionFeatures := ai.NewActionFeatures(b, action, 0)

	// Completely unrelated position should be zero.
	got := getPosition(Pos{-1, 0}, action.SourcePos, &actionFeatures.SourceFeatures)
	if !isZero(got) {
		glog.Errorf("Features for Pos{-1,0} should be 0, got %s instead",
			ai.PositionFeaturesToString(got))
	}

	// Check stacked pieces.
	// We want: [Top: Beetle(Current), Stack([-1 0 0]), Bottom: Queen(Opponent) / SrcTgt=0]
	pos := Pos{0, 1}
	want := []float32{1, 0, 1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0}
	got = getPosition(pos, action.SourcePos, &actionFeatures.SourceFeatures)
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Position %s with stack: Got %s, wanted %s\n",
			pos,
			ai.PositionFeaturesToString(got),
			ai.PositionFeaturesToString(want),
		)
	}

	// Check grasshopper before and after move.
	// Before in Pos{-1, -1}: want [Top: Grasshopper(Current), Stack([0 0 0]), Bottom: Grasshopper(Current) / SrcTgt=1]
	pos = Pos{-1, -1}
	want = []float32{1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0}
	got = getPosition(pos, action.SourcePos, &actionFeatures.SourceFeatures)
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Grasshopper in %s before move: Got %s, wanted %s\n",
			pos,
			ai.PositionFeaturesToString(got),
			ai.PositionFeaturesToString(want),
		)
	}
	// Before in Pos{1, 0}: want [Empty / SrcTgt=1]
	pos = Pos{1, 0}
	got = getPosition(pos, action.SourcePos, &actionFeatures.SourceFeatures)
	want = []float32{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Grasshopper in %s before move: Got %s, wanted %s\n",
			pos,
			ai.PositionFeaturesToString(got),
			ai.PositionFeaturesToString(want),
		)
	}
	// After in Pos{-1, -1}: want [Empty / SrcTgt=1]
	pos = Pos{-1, -1}
	got = getPosition(pos, action.TargetPos, &actionFeatures.TargetFeatures)
	want = []float32{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Grasshopper in %s after move: Got %s, wanted %s\n",
			pos,
			ai.PositionFeaturesToString(got),
			ai.PositionFeaturesToString(want),
		)
	}
	// After in Pos{1, 0}: want want [Top: Grasshopper(Current), Stack([0 0 0]), Bottom: Grasshopper(Current) / SrcTgt=1]
	pos = Pos{1, 0}
	got = getPosition(pos, action.TargetPos, &actionFeatures.TargetFeatures)
	want = []float32{1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0}
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Grasshopper in %s after move: Got %s, wanted %s\n",
			pos,
			ai.PositionFeaturesToString(got),
			ai.PositionFeaturesToString(want),
		)
	}
}

// Print boards with rotated neighborhood.
func debugNeighboursForPos(ui *ascii_ui.UI, base Pos) {
	neig := ai.X_EVEN_NEIGHBOURS
	if base.X()%2 == 1 {
		neig = ai.X_ODD_NEIGHBOURS
	}
	for rotation := 0; rotation < 6; rotation += 1 {
		b := NewBoard()
		for neigSlice := 0; neigSlice < 6; neigSlice++ {
			idx0 := (neigSlice + rotation) % 6
			for idx1 := 0; idx1 < 3; idx1++ {
				pos := Pos{base.X() + neig[idx0][idx1][0], base.Y() + neig[idx0][idx1][1]}
				fmt.Println(pos)
				b.StackPiece(pos, uint8(neigSlice%2), (Piece(neigSlice)%NUM_PIECE_TYPES)+ANT)
			}
		}
		b.BuildDerived()
		ui.Print(b)
	}
}
