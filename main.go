package main

import (
	"flag"
	"fmt"
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

	board := NewBoard()
	board.MaxMoves = *flag_maxMoves

	ui := ascii_ui.NewUI(true, false)
	_, err := ui.Run(board)
	if err != nil {
		log.Fatalf("Failed to run match: %v", err)
	}
	// ui.Print(board)
}
