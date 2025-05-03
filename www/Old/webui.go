package main

import (
	"fmt"
	"github.com/gopherjs/gopherjs/js"
	"github.com/gopherjs/jquery"
	"math"

	"github.com/gowebapi/webapi/graphics/svg"
	"github.com/gowebapi/webapi/html"
	"github.com/gowebapi/webapi/html/canvas"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/state"
)

// ZoomOnWheel responds to mouse wheel movement to control zoom.
func ZoomOnWheel(e jquery.Event) {
	wheelEvent := e.Object.Get("originalEvent")
	scrollAmount := wheelEvent.Get("deltaY").Float()
	ui.Scale = ui.Scale * math.Pow(1.1, scrollAmount/50.0)
	OnChangeOfUIParams()
}

// Variables that control the drag of the g.board.
var (
	DragStarted  = false
	DragX, DragY int
)

func DragOnMouseDown(e jquery.Event) {
	DragStarted = true
	DragX, DragY = e.PageX, e.PageY
}

func DragOnMouseUp(_ jquery.Event) {
	DragStarted = false
}

func DragOnMouseMove(e jquery.Event) {
	if DragStarted {
		deltaX, deltaY := e.PageX-DragX, e.PageY-DragY
		ui.ShiftX += float64(deltaX)
		ui.ShiftY += float64(deltaY)
		DragX, DragY = e.PageX, e.PageY
		OnChangeOfUIParams()
	}
}

// Place where available pieces are displayed.
var (
	BoardGroup     jquery.JQuery
	OffBoardGroups [2]jquery.JQuery
	OffBoardRects  [2]*js.Object
)

func (g *Game) createBoardRects() {
	BoardGroup = jq(CreateSVG("g", Attrs{
		"x":      0,
		"y":      0,
		"width":  "100%",
		"height": "100%",
	}))
	canvas.Append(BoardGroup)

	for ii := 0; ii < 2; ii++ {
		OffBoardGroups[ii] = jq(CreateSVG("g", Attrs{
			"x":      0,
			"y":      0,
			"width":  "100%",
			"height": "100%",
		}))
		canvas.Append(OffBoardGroups[ii])
		OffBoardRects[ii] = CreateSVG("rect", Attrs{
			"stroke": "firebrick",
			"fill":   "moccasin",
		})
		OffBoardGroups[ii].Append(OffBoardRects[ii])
	}
	g.MarkNextPlayer()
}

func (g *Game) MarkNextPlayer() {
	for player := range state.PlayerNum(state.NumPlayers) {
		width := 2.0 * ui.PixelRatio
		stroke := "firebrick"
		if g.board.IsFinished() {
			if g.board.Draw() || g.board.Winner() == player {
				stroke = "url(#colors)"
				width = 12 * ui.PixelRatio
			}
		} else {
			if g.board.NextPlayer == player {
				width = 6 * ui.PixelRatio
			}
		}
		SetAttrs(OffBoardRects[player], Attrs{
			"stroke-width": width,
			"stroke":       stroke,
		})
	}
}

var (
	StartGameDialog = struct {
		Div, Button jquery.JQuery
	}{
		jq("div#new_game"),
		jq("button#start"),
	}
)

func OpenStartGameDialog() {
	StartGameDialog.Div.SetCss("display", "block")
	StartGameDialog.Button.On(jquery.CLICK, func(e jquery.Event) {
		StartGameDialogDone()
	})
}

func StartGameDialogDone() {
	// TODO: clean and restart board and UI pieces.
	g := &Game{
		board:     state.NewBoard(),
		IsRunning: true,
	}

	// Prepare game.
	gameType := jq("input[name=game_type]:checked").Val()
	StartGameDialog.Div.SetCss("display", "none")
	if gameType == "ai" {
		aiConfig := jq("input[name=ai_config]").Val()
		fmt.Printf("AI: config=%s, starts=%s\n", aiConfig,
			jq("input[name=ai_starts]:checked").Val())
		aiPlayer := state.PlayerSecond
		if jq("input[name=ai_starts]:checked").Val() == "on" {
			aiPlayer = state.PlayerFirst
		}
		err := g.StartAI(aiConfig, aiPlayer)
		if err != nil {
			fmt.Printf("Failed to create an AI player:\n%+v\n", err)
			Alert(err.Error())
		}
	}
}

var (
	HasEndGameMessage = false
	EndGameMessage    jquery.JQuery
)

func (g *Game) ShowEndGameMessage(text string) {
	EndGameMessage = jq(CreateSVG("text", Attrs{
		"id":                 "end_game_message",
		"fill":               "url(#colors)",
		"class":              "end_game_message_class",
		"alignment-baseline": "middle",
		"text-anchor":        "middle",
		"pointer-events":     "none",
	}))
	EndGameMessage.Append(text)
	canvas.Append(EndGameMessage)
	HasEndGameMessage = true
	g.AdjustEndGameMessagePosition()
}

func (g *Game) AdjustEndGameMessagePosition() {
	if HasEndGameMessage {
		var y float64
		if g.board.Draw() {
			y = float64(ui.Height) / 2.0
		} else if g.board.Winner() == 0 {
			y = float64(ui.OffBoardHeight()) + 50*ui.PixelRatio
		} else {
			y = float64(ui.Height-ui.OffBoardHeight()) - 50*ui.PixelRatio
		}
		SetAttrs(Obj(EndGameMessage), Attrs{
			"x": ui.Width / 2.0,
			"y": y,
		})
	}
}

// Alert Window with given messate.
func Alert(msg string) {
	js.Global.Call("alert", msg)
}

func (g *Game) OnCanvasResize() {
	ui.Width = canvas.InnerWidth()
	ui.Height = canvas.InnerHeight()

	// OffBoard space.
	offboardHeight := ui.OffBoardHeight()
	if ui.Height < 3*offboardHeight {
		ui.Height = 3 * offboardHeight
	}
	SetAttrs(OffBoardRects[0], Attrs{
		"x": 0, "y": 0,
		"width":  ui.Width,
		"height": offboardHeight,
	})
	SetAttrs(OffBoardRects[1], Attrs{
		"x": 0, "y": ui.Height - offboardHeight,
		"width":  ui.Width,
		"height": offboardHeight,
	})
	g.MarkNextPlayer()
	Old.AdjustOffBoardPieces()
	AdjustSplashScreen()
	g.AdjustEndGameMessagePosition()
	Old.AdjustBusyBoxPosition()

	// Adjust all elements on page.
	OnChangeOfUIParams()
}

func OnChangeOfUIParams() {
	Old.PiecesOnChangeOfUIParams()
	Old.SelectionsOnChangeOfUIParams()
}

func Obj(jo jquery.JQuery) *js.Object {
	return jo.Underlying().Index(0)
}

// Game holds together the game information: current board, current player,
// and the AI player.
type Game struct {
	board     *state.Board
	IsRunning bool

	aiPlayer              *players.SearcherScorer
	IsAIPlaying, IsAITurn bool

	//idleChan allows cooperative goroutines to yield the focus.
	idleChan      chan bool
	isCooperative bool // Whether idleChan is being used.
}

func (g *Game) ExecuteAction(action state.Action) {
	// Animate action.
	if !action.IsSkipAction() {
		player := g.board.NextPlayer
		if action.Move {
			Old.RemovePiece(action)
		} else {
			Old.RemoveOffBoardPiece(player, action)
		}
		g.Place(player, action)
	}

	g.board = g.board.Act(action)
	if g.board.IsFinished() {
		g.MarkNextPlayer()
		var msg string
		if g.board.Draw() {
			msg = "Draw !!!"
		} else if g.board.Winner() == state.PlayerFirst {
			msg = "Top Player Wins !!!"
		} else {
			msg = "Bottom Player Wins !!!"
		}
		g.ShowEndGameMessage(msg)
		g.IsRunning = false
		return
	}

	if len(g.board.Derived.Actions) == 1 && g.board.Derived.Actions[0].IsSkipAction() {
		// Recurse to a skip action.
		g.ExecuteAction(g.board.Derived.Actions[0])
		return
	}

	// Select the next player.
	g.MarkNextPlayer()
	g.ScheduleAIPlay()
}

func main() {
	//show jQuery Version on console:
	print("Your current jQuery version is: " + jq().Jquery)

	// Create board parts.
	g := &Game{
		board:     state.NewBoard(),
		IsRunning: true,
	}

	// Create WebUI.
	ui = NewWebUI()
	g.createBoardRects()
	CreateSplashScreen()
	g.PlaceOffBoardPieces()
	g.OnCanvasResize()

	canvas.On("wheel", ZoomOnWheel)
	canvas.On(jquery.MOUSEDOWN, DragOnMouseDown)
	canvas.On(jquery.MOUSEUP, DragOnMouseUp)
	canvas.On(jquery.MOUSEMOVE, DragOnMouseMove)
	jq(Window).On(jquery.RESIZE, g.OnCanvasResize)

}
