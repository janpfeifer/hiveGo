package features_test

import (
	"fmt"
	features "github.com/janpfeifer/hiveGo/internal/features"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"k8s.io/klog/v2"
	"log"
	"reflect"
	"testing"
)

func printBoardAction(b *Board, action Action) {
	ui := cli.New(true, false)
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
func getPosition(pos, center Pos, f *features.ActionPositionFeatures) []float32 {
	relPos := Pos{pos.X() - center.X(), pos.Y() - center.Y()}
	if relPos.X() == 0 && relPos.Y() == 0 {
		return f.Center
	}
	neighbourhood := &features.X_EVEN_NEIGHBOURS
	if center.X()%2 != 0 {
		neighbourhood = &features.X_ODD_NEIGHBOURS
	}

	for section := 0; section < 6; section++ {
		for ii := 0; ii < features.POSITIONS_PER_SECTION; ii++ {
			neighPos := neighbourhood[section][ii]
			if relPos.Equal(neighPos) {
				return f.Sections[section][ii*int(features.IdPositionLast) : (ii+1)*int(features.IdPositionLast)]
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
	printBoardAction(b, action)
	actionFeatures := features.NewActionFeatures(b, action, 0)

	// Completely unrelated position should be zero.
	got := getPosition(Pos{-1, 0}, action.SourcePos, &actionFeatures.SourceFeatures)
	if !isZero(got) {
		klog.Errorf("Features for Pos{-1,0} should be 0, got %s instead",
			features.PositionFeaturesToString(got))
	}

	// Check stacked pieces.
	// We want: [Top: Beetle(Current), Stack([-1 0 0]), Bottom: Queen(Opponent) / SrcTgt=0]
	pos := Pos{0, 1}
	want := []float32{1, 0, 1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0}
	got = getPosition(pos, action.SourcePos, &actionFeatures.SourceFeatures)
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Position %s with stack: Got %s, wanted %s\n",
			pos,
			features.PositionFeaturesToString(got),
			features.PositionFeaturesToString(want),
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
			features.PositionFeaturesToString(got),
			features.PositionFeaturesToString(want),
		)
	}
	// Before in Pos{1, 0}: want [Empty / SrcTgt=1]
	pos = Pos{1, 0}
	got = getPosition(pos, action.SourcePos, &actionFeatures.SourceFeatures)
	want = []float32{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Grasshopper in %s before move: Got %s, wanted %s\n",
			pos,
			features.PositionFeaturesToString(got),
			features.PositionFeaturesToString(want),
		)
	}
	// After in Pos{-1, -1}: want [Empty / SrcTgt=1]
	pos = Pos{-1, -1}
	got = getPosition(pos, action.TargetPos, &actionFeatures.TargetFeatures)
	want = []float32{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Grasshopper in %s after move: Got %s, wanted %s\n",
			pos,
			features.PositionFeaturesToString(got),
			features.PositionFeaturesToString(want),
		)
	}
	// After in Pos{1, 0}: want want [Top: Grasshopper(Current), Stack([0 0 0]), Bottom: Grasshopper(Current) / SrcTgt=1]
	pos = Pos{1, 0}
	got = getPosition(pos, action.TargetPos, &actionFeatures.TargetFeatures)
	want = []float32{1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0}
	if !reflect.DeepEqual(want, got) {
		fmt.Printf("Grasshopper in %s after move: Got %s, wanted %s\n",
			pos,
			features.PositionFeaturesToString(got),
			features.PositionFeaturesToString(want),
		)
	}
}

// Print boards with rotated neighborhood.
func debugNeighboursForPos(ui *cli.UI, base Pos) {
	neig := features.X_EVEN_NEIGHBOURS
	if base.X()%2 == 1 {
		neig = features.X_ODD_NEIGHBOURS
	}
	for rotation := 0; rotation < 6; rotation += 1 {
		b := NewBoard()
		for neigSlice := 0; neigSlice < 6; neigSlice++ {
			idx0 := (neigSlice + rotation) % 6
			for idx1 := 0; idx1 < 3; idx1++ {
				pos := Pos{base.X() + neig[idx0][idx1][0], base.Y() + neig[idx0][idx1][1]}
				fmt.Println(pos)
				b.StackPiece(pos, PlayerNum(neigSlice%2), (PieceType(neigSlice)%NumPieceTypes)+ANT)
			}
		}
		b.BuildDerived()
		ui.Print(b, false)
	}
}
