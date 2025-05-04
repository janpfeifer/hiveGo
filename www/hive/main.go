package main

import (
	"fmt"
	"github.com/gowebapi/webapi/backgroundtask"

	"github.com/gowebapi/webapi"
	"github.com/gowebapi/webapi/core/js"
	"github.com/gowebapi/webapi/html/htmlevent"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/simplego"
	_ "github.com/janpfeifer/hiveGo/internal/players/default"
)

// HTML page core elements:
var (
	Document = webapi.GetDocument()
	Window   = webapi.GetWindow()
)

func main() {
	ui := NewWebUI()
	fmt.Printf("UI: %+v\n", ui)
	gameDialog := func() {
		ui.OpenGameStartDialog(func() { _ = NewGame(ui) })
	}
	ui.CreateSplashScreen(gameDialog)
	Window.SetOnResize(func(event *htmlevent.UIEvent, currentTarget *webapi.Window) {
		ui.OnCanvasResize()
	})
	//ui.SetBusy(true)

	// Wait forever: the Wasm program will never exit, while the page is opened.
	select {}
}

func startGame(ui *WebUI) {
}

// Game coordinates the execution of the game.
type Game struct {
	board            *state.Board
	ui               *WebUI
	aiPlayer         *players.SearcherScorer
	aiPlayerNum      state.PlayerNum
	idleChan         chan bool
	idleCallbackFunc js.Func
}

// NewGame creates and starts a new game using the provided UI.
func NewGame(ui *WebUI) *Game {
	fmt.Printf("hotseat=%v, aiStarts=%v, aiConfig=%q\n", ui.IsHotseat(), ui.AIStarts(), ui.gameStartAIConfig.Value())
	g := &Game{
		board:    state.NewBoard(),
		ui:       ui,
		idleChan: make(chan bool, 1),
	}
	g.ui.StartBoard(g.board)
	var err error
	g.aiPlayer, err = players.New(ui.gameStartAIConfig.Value())
	if err != nil {
		msg := fmt.Sprintf("Failed to created: %+v\n\nReload to start again.", err)
		klog.Error(msg)
		Window.Alert2(msg)
		klog.Fatal("Stopping game.")
	}
	if ui.AIStarts() {
		g.aiPlayerNum = state.PlayerFirst
	} else {
		g.aiPlayerNum = state.PlayerSecond
	}

	// Setup "cooperative" concurrency required by the browser.
	g.idleCallbackFunc = js.FuncOf(func(js.Value, []js.Value) interface{} {
		g.idleCallback()
		return nil
	})
	Window.RequestIdleCallback((*backgroundtask.IdleRequestCallback)(&g.idleCallbackFunc), nil)
	g.aiPlayer.Searcher.SetCooperative(func() {
		<-g.idleChan // Read one element, which triggers yielding processing back to the browser.
	})

	go g.RunGame()
	return g
}

// idleCallback is called everytime the browser UI is idle and we can do some amount of processing.
func (g *Game) idleCallback() {
	// Send a signal to process a chunk but doesn't block.
	select {
	case g.idleChan <- true:
		// Process a chunk.
	default:
		// Nothing to do: nobody is listening on the other side.
	}
	Window.RequestIdleCallback((*backgroundtask.IdleRequestCallback)(&g.idleCallbackFunc), nil)
}

func (g *Game) RunGame() {
	ui := g.ui
	for !g.board.IsFinished() {
		// Cooperative concurrency with the browser, let the UI catch up.
		<-g.idleChan

		var nextBoard *state.Board
		var action state.Action
		if g.board.NextPlayer == g.aiPlayerNum || true {
			action, nextBoard, _, _ = g.aiPlayer.Play(g.board)

		} else {
			// User play:
			klog.Error("not implemented")
			select {}

		}

		// Update the UI accordingly.
		if !action.IsSkipAction() {
			player := g.board.NextPlayer
			if action.Move {
				ui.RemoveOnBoardPiece(action)
			} else {
				ui.RemoveOffBoardPiece(player, action)
			}
			ui.PlaceOnBoardPiece(player, action)
		}
		g.board = nextBoard
		ui.UpdateBoard(g.board)
	}
	fmt.Printf("Game finished: %s won!\n", g.board.Winner())
}
