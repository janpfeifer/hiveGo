package main

import (
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/ai"
	"log"

	// TensorFlow is included so it shows up as an option for scorers.
	_ "github.com/janpfeifer/hiveGo/ai/tensorflow"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = fmt.Printf

var flag_players = [2]*string{
	flag.String("p0", "ai", "First player: hotseat, online, ai"),
	flag.String("p1", "ai", "First player: hotseat, online, ai"),
}
var flag_aiUI = flag.Bool("ai_ui", true, "Shows UI even for ai vs ai game.")
var flag_maxMoves = flag.Int(
	"max_moves", 200, "Max moves before game is assumed to be a draw.")

func main() {
	flag.Parse()
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}

	if true {
		debugNeighbours()
		return
	}

	board := NewBoard()
	board.MaxMoves = *flag_maxMoves

	ui := ascii_ui.NewUI(true, false)
	_, err := ui.Run(board)
	if err != nil {
		log.Fatalf("Failed to run match: %v", err)
	}
	// ui.Print(board)
}

func debugNeighbours() {
	ui := ascii_ui.NewUI(true, false)
	debugNeighboursForPos(ui, Pos{0, 0})
	debugNeighboursForPos(ui, Pos{1, 0})
}

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
