package state_test

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	. "github.com/janpfeifer/hiveGo/internal/state/statetest"
	"github.com/stretchr/testify/require"
	"log"
	"strings"
	"testing"
)

var (
	testBoards = []string{
		// Simple double loop
		`
R.R
.N.
R.R`,

		// Large loop
		`
...
.R.
R.R
.R.
R.R
.R.
`,
		// Large loop with branch.
		`
...
.R.
R.R.N.N.R
.R.N.N.N.
R.R
.R.
`,

		// Test used for benchmark.
		benchmarkBoardText,
	}

	benchmarkBoardText = `
..........
...R......
R.........
.N.N.R.N.R
..N.N.N.N.
.....R...R
..N.R.....
.R........
..R.......
.R........
`
)

func convertTextToBoard(txt string) (b *Board, removable generics.Set[Pos]) {
	lines := strings.Split(txt, "\n")
	if lines[0] == "" {
		lines = lines[1:]
	}

	b = NewBoard()
	removable = generics.MakeSet[Pos]()

	for row, line := range lines {
		for col, code := range line {
			if (row+col)%2 == 1 && code != '.' {
				log.Panicf("Board at row %d, col %d should be '.', is '%c' instead.\n%s", row, col, code, txt)
			}
			if code == '.' {
				continue
			}
			x := col
			y := row>>1 - x>>1
			pos := Pos{int8(col), int8(y)}

			b.StackPiece(pos, 0, ANT)
			if code == 'R' {
				removable.Insert(pos)
			} else if code != 'N' {
				log.Panicf("Board at row %d, col %d unexpected code '%c'.\n%s", row, col, code, txt)
			}
		}
	}

	b.BuildDerived()
	return
}

// testRemovableAlternatives was created for testing different variations of the algorithm.
// We left only the fastest, but this function is left here as is if one day one tries
// a different one.
func testRemovableAlternatives(t *testing.T, version int) {
	fmt.Printf("Testing removable version %d:\n", version)
	for boardIdx, txt := range testBoards {
		board, want := convertTextToBoard(txt)
		fmt.Printf("> Board #%d:\n%s\n\n", boardIdx, txt)
		PrintBoard(board)
		startIdx := 0
		for start := range board.OccupiedPositionsIter() {
			var removable generics.Set[Pos]
			switch version {
			case 0:
				removable = board.RemovablePositions(start)
			}
			require.Truef(t, want.Equal(removable), "Removable(version=%d, start#%d=%s):\n\twant=%v\n\t got=%v", version, startIdx, start, want, removable)
			startIdx++
		}
	}
}

func TestRemovable(t *testing.T) {
	testRemovableAlternatives(t, 0)
}

func BenchmarkRemovablePositions(b *testing.B) {
	board, _ := convertTextToBoard(benchmarkBoardText)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = board.RemovablePositions()
	}
}
