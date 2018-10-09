package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/golang/glog"
	"github.com/gotk3/gotk3/glib"
	"github.com/gotk3/gotk3/gtk"
	"github.com/janpfeifer/hiveGo/ai/players"
	"github.com/janpfeifer/hiveGo/ai/tensorflow"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = fmt.Printf

var (
	flag_players = [2]*string{
		flag.String("p0", "hotseat", "First player: hotseat, ai"),
		flag.String("p1", "hotseat", "Second player: hotseat, ai"),
	}
	flag_aiConfig = flag.String("ai", "", "Configuration string for the AI.")
	flag_maxMoves = flag.Int(
		"max_moves", 200, "Max moves before game is assumed to be a draw.")

	// TODO: find directory automatically basaed on GOPATH.
	flag_resources = flag.String("resources", "", "Directory with resources. "+
		"If empty it will try to search in GOPATH for the directory.")

	// Save match at end.
	flag_saveMatch = flag.String("save", "", "File name where to save match. Matches are appendeded to given file.")

	// Sequence of boards that make up for the game. Used for undo-ing actions.
	gameSeq []*Board
)

func init() {
	flag.BoolVar(&tensorflow.CpuOnly, "cpu", false, "Force to use CPU, even if GPU is available")
}

const APP_ID = "com.github.janpfeifer.hiveGo.gnome-hive"

// Board in use. It will always be set.
var (
	initial, board *Board
	actions        []Action
	scores         []float32
	started        bool // Starts as false, and set to true once a game is running.

	// Hints for the UI.
	finished  bool
	aiPlayers = [2]players.Player{nil, nil}
	nextIsAI  bool
)

func findResourcesDir() {
	if *flag_resources != "" {
		return
	}
	for _, p := range strings.Split(os.Getenv("GOPATH"), ":") {
		if p == "" {
			continue
		}
		p = p + "/src/github.com/janpfeifer/hiveGo/images"
		log.Printf("Looking at %s", p)
		s, err := os.Stat(p)
		if os.IsNotExist(err) {
			continue
		}
		if s.IsDir() {
			log.Printf("Found image resources in '%s'", p)
			*flag_resources = p
			return
		}
	}
	log.Fatal("Can't find location of image resources using ${GOPATH}, please " +
		"set it with --resources")
	return
}

// Open file for appending.
func openForAppending(filename string) io.WriteCloser {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Panicf("Failed to save file to '%s': %v", filename, err)
	}
	return file
}

func main() {
	flag.Parse()
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}
	findResourcesDir()

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
	initial = board
	actions = nil
	scores = nil
	gameSeq := make([]*Board, 0, *flag_maxMoves)
	gameSeq = append(gameSeq, board)

	// Create players:
	for ii := 0; ii < 2; ii++ {
		switch {
		case *flag_players[ii] == "hotseat":
			continue
		case *flag_players[ii] == "ai":
			aiPlayers[ii] = players.NewAIPlayer(*flag_aiConfig, true)
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
		action, _, _ := aiPlayers[board.NextPlayer].Play(board)
		executeAction(action)
	}
}

func executeAction(action Action) {
	glog.Infof("Player %d played %s", board.NextPlayer, action)
	board = board.Act(action)
	actions = append(actions, action)
	scores = append(scores, 0)
	gameSeq = append(gameSeq, board)
	finished = board.IsFinished()
	if !finished && len(board.Derived.Actions) == 0 {
		// Player has no available moves, skip.
		log.Printf("No action available, automatic action.")
		if action.Piece == NO_PIECE {
			// Two skip actions in a row.
			log.Fatal("No moves avaialble to either players !?")
		}
		// Recurse to a skip action.
		executeAction(Action{Piece: NO_PIECE})
		return
	}
	followAction()

	if board.IsFinished() && *flag_saveMatch != "" {
		log.Printf("Saving match to %s", *flag_saveMatch)
		file := openForAppending(*flag_saveMatch)
		enc := gob.NewEncoder(file)
		if err := SaveMatch(enc, initial.MaxMoves, actions, scores); err != nil {
			log.Printf("Failed to save match to %s: %v", *flag_saveMatch, err)
		}
		file.Close()
	}
}

func undoAction() {
	// Can't undo until it's human turn. TODO: add support for interrupting
	// AI.
	if nextIsAI || finished || len(gameSeq) < 2 {
		return
	}
	gameSeq = gameSeq[0 : len(gameSeq)-2]
	board = gameSeq[len(gameSeq)-1]
	followAction()
}

// Setting that come after executing an action.
func followAction() {
	selectedOffBoardPiece = NO_PIECE
	hasSelectedPiece = false
	nextIsAI = !finished && aiPlayers[board.NextPlayer] != nil
	if nextIsAI {
		// Start AI thinking on a separate thread.
		go func() {
			action, _, _ := aiPlayers[board.NextPlayer].Play(board)
			glib.IdleAdd(func() { executeAction(action) })
		}()
	}
	mainWindow.QueueDraw()
}
