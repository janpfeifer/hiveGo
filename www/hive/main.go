package main

import (
	"fmt"
	"runtime"
	"time"

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
	jsFunc := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		goPrintStacks()
		return nil
	})
	js.Global().Set("goPrintStacks", jsFunc)

	ui := NewWebUI()
	gameDialog := func() {
		ui.OpenGameStartDialog(func() { _ = NewGame(ui) })
	}
	ui.CreateSplashScreen(gameDialog)
	Window.SetOnResize(func(event *htmlevent.UIEvent, currentTarget *webapi.Window) {
		ui.OnCanvasResize()
	})

	// The Wasm program never exits, the main goroutine just goes to sleep.
	select {}
}

func startGame(ui *WebUI) {
}

// Game coordinates the execution of the game.
type Game struct {
	board       *state.Board
	ui          *WebUI
	aiPlayer    *players.SearcherScorer
	aiPlayerNum state.PlayerNum

	idleChan         chan bool
	idleCallbackFunc js.Func

	lastYield      time.Time
	yieldStartTime time.Time
	yieldCount     int
}

// NewGame creates and starts a new game using the provided UI.
func NewGame(ui *WebUI) *Game {
	klog.Infof("NewGame(): hotseat=%v, aiStarts=%v, aiConfig=%q\n", ui.IsHotseat(), ui.AIStarts(), ui.gameStartAIConfig.Value())
	g := &Game{
		board:      state.NewBoard(),
		ui:         ui,
		idleChan:   make(chan bool, 1),
		yieldCount: -1, // Not counting.
	}
	g.ui.StartBoard(g.board)
	if !ui.IsHotseat() {
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

		g.aiPlayer.Searcher.SetCooperative(func() {
			now := time.Now()

			// Count yields (each corresponds to one "eval" of the searcher.
			if g.yieldCount >= 0 {
				g.yieldCount++
				if elapsed := now.Sub(g.yieldStartTime); elapsed > time.Second {
					evalRate := float64(g.yieldCount) / elapsed.Seconds()
					g.ui.UpdateAIEvalRate(evalRate)
					g.yieldCount = 0
					g.yieldStartTime = now
				}
			}

			// Only yields every 20 ms.
			if now.Sub(g.lastYield) < 10*time.Millisecond {
				return
			}
			g.lastYield = now

			// Micro-sleeps: ugly ... but I don't know a better way to do this.
			// Waiting for a RequestIdleCallback callback stops working after ~1000 calls to it, not sure
			// why (maybe Go runtime for WebAssembly uses it in way the interferes...).
			time.Sleep(500 * time.Microsecond)
		})
	}

	go g.RunGame()
	return g
}

func (g *Game) RunGame() {
	ui := g.ui
	for !g.board.IsFinished() {
		var nextBoard *state.Board
		var action state.Action
		if g.aiPlayer != nil && g.board.NextPlayer == g.aiPlayerNum {
			ui.HideTutorial()
			g.yieldStartTime = time.Now()
			g.yieldCount = 0
			action, nextBoard, _, _ = g.aiPlayer.Play(g.board)
			if elapsed := time.Since(g.yieldStartTime); elapsed < time.Second {
				evalRate := float64(g.yieldCount) / elapsed.Seconds()
				g.ui.UpdateAIEvalRate(evalRate)
			}
			g.yieldCount = -1

		} else {
			action, nextBoard = g.UserPlay()
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

		// Update clocks and account time to the correct player.
		ui.UpdateTime()

		g.board = nextBoard
		ui.UpdateBoard(g.board)
	}

	ui.HideTutorial()
	if g.board.Winner() == state.PlayerInvalid {
		ui.SetWinner(fmt.Sprintf("Draw! No winner after %d moves.", g.board.MoveNumber))
	} else {
		ui.SetWinner(fmt.Sprintf("Player %d wins, congratulations!", g.board.Winner()+1))
	}
	fmt.Printf("Game finished: %s won!\n", g.board.Winner())
}

func (g *Game) UserPlay() (state.Action, *state.Board) {
	var action state.Action
	if g.board.NumActions() == 1 && g.board.Derived.Actions[0].IsSkipAction() {
		// Handle the case where there are no valid moves.
		currentPlayer := g.board.NextPlayer + 1
		Window.Alert2(fmt.Sprintf("Player %d has no valid moves, skipping back to player %d", currentPlayer, 3-currentPlayer))
		action = g.board.Derived.Actions[0]
	} else {
		// Action selected by the UI:
		action = g.ui.SelectAction()
	}
	return action, g.board.Act(action)
}

// goPrintStacks prints in the console all stacktraces.
func goPrintStacks() {
	buf := make([]byte, 20*1024)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			fmt.Print(string(buf[:n]))
			break
		}
		buf = make([]byte, 2*len(buf))
	}
}
