package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/gotk3/gotk3/glib"
	"github.com/gotk3/gotk3/gtk"
	"github.com/janpfeifer/hiveGo/internal/players"

	//"github.com/janpfeifer/hiveGo/ai/players"
	//_ "github.com/janpfeifer/hiveGo/ai/search/ab"
	//_ "github.com/janpfeifer/hiveGo/ai/search/mcts"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"io"
	"k8s.io/klog/v2"
	"log"
	"os"
	//"github.com/janpfeifer/hiveGo/ai/tensorflow"
	_ "github.com/janpfeifer/hiveGo/ai"
)

var _ = fmt.Printf

var (
	//flag_players = [2]*string{
	//	flag.String("p0", "hotseat", "First player: hotseat, ai"),
	//	flag.String("p1", "hotseat", "Second player: hotseat, ai"),
	//}
	flagAIConfig = flag.String("ai", "", "Configuration string for the AI.")
	flagMaxMoves = flag.Int(
		"max_moves", DefaultMaxMoves, "Max moves before game is considered a draw.")

	// Save match at end.
	flagSaveMatch = flag.String("save", "", "File name where to save match. Matches are appended to given file.")

	// Sequence of boards that make up for the game. Used for undo-ing actions.
	gameSeq []*Board
)

const AppId = "com.github.janpfeifer.hiveGo.cmd.gnome-hive"

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

// Open file for appending.
func openForAppending(filename string) io.WriteCloser {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Panicf("Failed to save file to %q: %v", filename, err)
	}
	return file
}

func main() {
	flag.Parse()
	if *flagMaxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flagMaxMoves)
	}

	// Build initial board: it is used only for drawing available pieces,
	board = NewBoard()
	board.MaxMoves = *flagMaxMoves
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
	board.MaxMoves = *flagMaxMoves
	board.BuildDerived()
	initial = board
	actions = nil
	scores = nil
	gameSeq := make([]*Board, 0, *flagMaxMoves)
	gameSeq = append(gameSeq, board)

	//// Create players:
	//for ii := 0; ii < 2; ii++ {
	//	switch {
	//	case *flag_players[ii] == "hotseat":
	//		continue
	//	case *flag_players[ii] == "ai":
	//		aiPlayers[ii] = players.New(*flagAIConfig, true)
	//	default:
	//		log.Fatalf("Unknown player type --p%d=%s", ii, *flag_players[ii])
	//	}
	//}

	// Initialize UI state.
	started = true
	finished = false
	zoomFactor = 1.
	shiftX, shiftY = 0., 0.
	mainWindow.QueueDraw()

	// AI starts playing ?
	if aiPlayers[board.NextPlayer] != nil {
		action, _, _, _ := aiPlayers[board.NextPlayer].Play(board, "AIvsHuman")
		executeAction(action)
	}
}

func executeAction(action Action) {
	klog.Infof("Player %d played %s", board.NextPlayer, action)
	board = board.Act(action)
	actions = append(actions, action)
	scores = append(scores, 0)
	gameSeq = append(gameSeq, board)
	finished = board.IsFinished()
	if !finished && len(board.Derived.Actions) == 0 {
		// Player has no available moves, skip.
		klog.Infof("No action available, automatic action.")
		if action.IsSkipAction() {
			// Two skip actions in a row.
			log.Fatal("No moves avaialble to either players !?")
		}
		// Recurse to a skip action.
		executeAction(SkipAction)
		return
	}
	followAction()

	if board.IsFinished() && *flagSaveMatch != "" {
		klog.Infof("Saving match to %s", *flagSaveMatch)
		file := openForAppending(*flagSaveMatch)
		defer func() { _ = file.Close() }()
		enc := gob.NewEncoder(file)
		if err := SaveMatch(enc, initial.MaxMoves, actions, scores, nil); err != nil {
			klog.Errorf("Failed to save match to %s: %v", *flagSaveMatch, err)
			klog.Errorf("Game continuing anyway...")
		}
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
	selectedOffBoardPiece = NoPiece
	hasSelectedPiece = false
	nextIsAI = !finished && aiPlayers[board.NextPlayer] != nil
	if nextIsAI {
		// Start AI thinking on a separate thread.
		go func() {
			action, _, _, _ := aiPlayers[board.NextPlayer].Play(board, "AIvsHuman")
			glib.IdleAdd(func() { executeAction(action) })
		}()
	}
	mainWindow.QueueDraw()
}
