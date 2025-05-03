package main

import (
	"fmt"
	"github.com/gowebapi/webapi"
	"github.com/gowebapi/webapi/html/htmlevent"
	"github.com/janpfeifer/hiveGo/internal/state"
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
		ui.OpenGameStartDialog(func() { startGame(ui) })
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
	fmt.Printf("hotseat=%v, aiStarts=%v, aiConfig=%q\n", ui.IsHotseat(), ui.AIStarts(), ui.gameStartAIConfig.Value())
	board := state.NewBoard()
	ui.EnableBoard()
	ui.CreateBoardRects(board)
}
