package main

import (
	"flag"
	"fmt"
	"log"
	"os"

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
var flag_resources = flag.String("resources",
	fmt.Sprintf("/home/%s/src/go/src/github.com/janpfeifer/hiveGo/images", os.Getenv("USER")),
	"Directory with resources")

const APP_ID = "com.github.janpfeifer.hiveGo.gnome-hive"

// Board in use. It will always be set.
var (
	board    *Board
	started  bool // Starts as false, and set to true once a game is running.
	finished bool
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
	finished = false
	zoomFactor = 1.
	shiftX, shiftY = 0., 0.
	mainWindow.QueueDraw()
}

func executeAction(action Action) {
	board = board.Act(action)
	board.BuildDerived()
	if len(board.Derived.Actions) == 0 {
		// Player has no available moves, skip.
		board = board.Act(Action{Piece: NO_PIECE})
		board.BuildDerived()
	}
	if board.Derived.Wins[0] || board.Derived.Wins[1] {
		finished = true
	}
	selectedOffBoardPiece = NO_PIECE
	hasSelectedPiece = false
	mainWindow.QueueDraw()
}
