package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/gotk3/gotk3/gtk"
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

// TODO: Find out automatically where resources are installed.
var flag_resources = flag.String("resources", "/home/janpf/src/go/src/github.com/janpfeifer/hiveGo/images",
	"Directory with resources")

const APP_ID = "com.github.janpfeifer.hiveGo.gnome-hive"

// Board in use. It will always be set.
var (
	board   *Board
	started bool // Starts as false, and set to true once a game is running.
)

func main() {
	flag.Parse()
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}
	board = NewBoard()
	board.MaxMoves = *flag_maxMoves
	board.BuildDerived()

	gtk.Init(nil)
	createMainWindow()
	mainWindow.ShowAll()
	gtk.Main()
}

func newGame() {
	board = NewBoard()
	board.MaxMoves = *flag_maxMoves
	board.BuildDerived()
	started = true
	mainWindow.QueueDraw()
}
