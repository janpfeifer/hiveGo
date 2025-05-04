package main

import (
	"fmt"
	"github.com/gowebapi/webapi"
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
	board       *state.Board
	ui          *WebUI
	aiPlayer    players.Player
	aiPlayerNum state.PlayerNum
}

// NewGame creates and starts a new game using the provided UI.
func NewGame(ui *WebUI) *Game {
	fmt.Printf("hotseat=%v, aiStarts=%v, aiConfig=%q\n", ui.IsHotseat(), ui.AIStarts(), ui.gameStartAIConfig.Value())
	g := &Game{
		board: state.NewBoard(),
		ui:    ui,
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

	go g.RunGame()
	return g
}

func (g *Game) RunGame() {
	ui := g.ui
	for !g.board.IsFinished() {
		var nextBoard *state.Board
		var action state.Action
		if g.board.NextPlayer == g.aiPlayerNum {
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
}
