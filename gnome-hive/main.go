package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/gotk3/gotk3/gtk"
	"github.com/janpfeifer/hiveGo/ai/players"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = fmt.Printf

var flag_players = [2]*string{
	flag.String("p0", "hotseat", "First player: hotseat, online, ai"),
	flag.String("p1", "hotseat", "Second player: hotseat, online, ai"),
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
	board     *Board
	started   bool // Starts as false, and set to true once a game is running.
	finished  bool
	aiPlayers = [2]players.Player{nil, nil}
	nextIsAI  bool
)

func main() {
	flag.Parse()
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}

	// Build initial board: it is used only for drawing available pieces,
	board = NewBoard()
	board.MaxMoves = *flag_maxMoves
	board.BuildDerived()

	// Creates and runs main window.
	gtk.Init(nil)
	createMainWindow()
	mainWindow.ShowAll()
	gtk.Main()
}

func newGame() {
	// Create board.
	board = NewBoard()
	board.MaxMoves = *flag_maxMoves
	board.BuildDerived()

	// Create players:
	for ii := 0; ii < 2; ii++ {
		switch {
		case *flag_players[ii] == "hotseat":
			continue
		case *flag_players[ii] == "ai":
			aiPlayers[ii] = players.NewAIPlayer()
		default:
			log.Fatalf("Unknown player type --p%d=%s", ii, *flag_players[ii])
		}
	}

	// Initialize UI state.
	started = true
	finished = false
	zoomFactor = 1.
	shiftX, shiftY = 0., 0.
	mainWindow.QueueDraw()

	// AI starts playing ?
	if aiPlayers[board.NextPlayer] != nil {
		action := aiPlayers[board.NextPlayer].Play(board)
		executeAction(action)
	}
}

func executeAction(action Action) {
	board = board.Act(action)
	if len(board.Derived.Actions) == 0 {
		// Player has no available moves, skip.
		log.Printf("No action available, automatic action.")
		board = board.Act(Action{Piece: NO_PIECE})
		if len(board.Derived.Actions) == 0 {
			log.Fatal("No moves avaialble to either players !?")
		}
	}

	finished = board.IsFinished()
	selectedOffBoardPiece = NO_PIECE
	hasSelectedPiece = false
	nextIsAI = !finished && aiPlayers[board.NextPlayer] != nil
	if nextIsAI {
		// Start AI thinking on a separate thread.
		go func() {
			action = aiPlayers[board.NextPlayer].Play(board)
			executeAction(action)
		}()
	}
	mainWindow.QueueDraw()
}
