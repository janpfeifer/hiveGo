package state_test

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/generics"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"log"
	"reflect"
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

func convertTextToBoard(txt string) (b *Board, removable map[Pos]bool, start Pos) {
	lines := strings.Split(txt, "\n")
	if lines[0] == "" {
		lines = lines[1:]
	}

	b = NewBoard()
	removable = make(map[Pos]bool, len(lines))

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
			start = pos

			b.StackPiece(pos, 0, ANT)
			if code == 'R' {
				removable[pos] = true
			} else if code != 'N' {
				log.Panicf("Board at row %d, col %d unexpected code '%c'.\n%s", row, col, code, txt)
			}
		}
	}

	b.BuildDerived()
	return
}

func testRemovableAlternatives(t *testing.T, useOld bool) {
	for _, txt := range testBoards {
		board, want, start := convertTextToBoard(txt)
		fmt.Println()
		fmt.Println(txt)
		fmt.Println()
		printBoard(board)
		fmt.Println()
		var removable generics.Set[Pos]
		if useOld {
			removable = board.OldRemovablePositions()
		} else {
			removable = board.TestRemovablePositionsForPos(start)
		}
		if !reflect.DeepEqual(removable, want) {
			t.Errorf("Removable(%v): wanted=%v, got=%v", useOld, want, removable)
		}
	}
}

func TestRemovable(t *testing.T) {
	testRemovableAlternatives(t, false)
}

func TestOldRemovable(t *testing.T) {
	testRemovableAlternatives(t, true)
}

func BenchmarkRemovablePositions(b *testing.B) {
	board, _, start := convertTextToBoard(benchmarkBoardText)
	//fmt.Println()
	//printBoard(board)
	//fmt.Println()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = board.TestRemovablePositionsForPos(start)
	}
}

func BenchmarkOldRemovablePositions(b *testing.B) {
	board, _, _ := convertTextToBoard(benchmarkBoardText)
	//fmt.Println()
	//printBoard(board)
	//fmt.Println()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = board.OldRemovablePositions()
	}
}
