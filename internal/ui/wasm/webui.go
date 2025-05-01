package main

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/players"
	"math"

	"github.com/gopherjs/gopherjs/js"
	"github.com/gopherjs/jquery"
	"github.com/janpfeifer/hiveGo/internal/state"
)

// Convenience:
var jq = jquery.NewJQuery

const (
	SVGCanvas = "svg#canvas"
	SVGDefs   = "defs#svgdefs"

	StandardFace = 33.0
)

var (
	window   = js.Global
	document = js.Global.Get("document")
	Canvas   = jq(SVGCanvas)
	SvgDefs  = jq(SVGDefs)
)

type UIParams struct {
	// PixelRatio gives a sense of how dense are pixels, where
	// 1.0 is "standard". It affects the zoom level of pieces off-g.board.
	PixelRatio float64

	// UI size: typically the window size, but could be
	// larger, in case the window is too little to fit everythin,
	// in which case things will be drawn off-screen.
	Width, Height int

	// Scale is controlled by the mouse wheel. It affects
	// only the board, not the pieces offg.board.
	Scale float64

	// How much the board has been dragged around.
	ShiftX, ShiftY float64
}

// ui contains all the current display parameters of the UI.
var ui *UIParams

func NewUIParams() *UIParams {
	ui := &UIParams{
		// PixelRatio should be set to window.Get("devicePixelRatio").Float(),
		// but at least on PixelBook, the browser already zoom things to
		// the PixelRatio without us needing to do anything. So fixed to 1.
		// for now.
		PixelRatio: 1.0,
		Width:      Canvas.InnerWidth(),
		Height:     Canvas.InnerHeight(),
		Scale:      window.Get("devicePixelRatio").Float(),
		ShiftX:     0,
		ShiftY:     0,
	}
	return ui
}

// Face returns the size of a face of an hexagon for the
// UI configuration.
func (ui *UIParams) Face() float64 {
	return StandardFace * ui.Scale
}

// OffBoardHeight for UI.
func (ui *UIParams) OffBoardHeight() int {
	return int(128.0 * ui.PixelRatio)
}

// hexTriangleHeight returns the height of the triangles that make up for an hexagon, given the face lenght.
func hexTriangleHeight(face float64) float64 {
	return 0.866 * face // sqrt(3)/2 * face
}

// PosToXY converts board positions to screen position.
func (ui *UIParams) PosToXY(pos state.Pos, stackCount int) (x, y float64) {
	face := ui.Face()
	hexWidth := 1.5 * face
	triangleHeight := hexTriangleHeight(face)
	hexHeight := 2. * triangleHeight

	x = ui.ShiftX + float64(pos.X())*hexWidth
	y = ui.ShiftY + float64(pos.Y())*hexHeight
	if pos.X()%2 != 0 {
		y += triangleHeight
	}
	x += float64(stackCount) * 3.0 * ui.Scale
	y -= float64(stackCount) * 3.0 * ui.Scale

	// Add center of screen:
	x += float64(ui.Width / 2)
	y += float64(ui.Height / 2)
	return
}

type Attrs map[string]interface{}

func SetAttrs(elem *js.Object, attrs Attrs) {
	for key, value := range attrs {
		elem.Call("setAttributeNS", nil, key, value)
	}
}

func CreateSVG(elemType string, attrs Attrs) *js.Object {
	elem := document.Call("createElementNS", "http://www.w3.org/2000/svg", elemType)
	SetAttrs(elem, attrs)
	return elem
}

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
	Canvas.Append(BoardGroup)

	for ii := 0; ii < 2; ii++ {
		OffBoardGroups[ii] = jq(CreateSVG("g", Attrs{
			"x":      0,
			"y":      0,
			"width":  "100%",
			"height": "100%",
		}))
		Canvas.Append(OffBoardGroups[ii])
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
	// References to splash screen objects created at the beginning of the program.

	HasSplashScreen                        = false
	SplashRect, SplashPattern, SplashImage jquery.JQuery
)

func CreateSplashScreen() {
	SplashPattern = jq(CreateSVG("pattern", Attrs{
		"id":           "splash",
		"patternUnits": "objectBoundingBox",
		"width":        "1.0",
		"height":       "1.0",
		"x":            "-0.040",
		"y":            "0",
	}))
	SplashImage = jq(CreateSVG("image", Attrs{
		"href":   "/github.com/janpfeifer/hiveGo/images/Grasshopper.png",
		"width":  1024,
		"height": 1024,
	}))
	SvgDefs.Append(SplashPattern)
	SplashPattern.Append(SplashImage)
	SplashRect = jq(CreateSVG("rect", Attrs{
		"stroke":       "black",
		"stroke-width": 0,
		"border":       0,
		"padding":      0,
		"fill":         "url(#splash)",
	}))
	Canvas.Append(SplashRect)
	SplashRect.On(jquery.MOUSEUP, func(e jquery.Event) {
		RemoveSplashScreen()
		OpenStartGameDialog()
	})
	HasSplashScreen = true
	AdjustSplashScreen()
}

func AdjustSplashScreen() {
	if !HasSplashScreen {
		return
	}

	// SplashScreen will have at most 1024 "face" size. It's a square
	// so this will be the width and height.
	face := ui.Height - 2*ui.OffBoardHeight()
	if ui.Width < face {
		face = ui.Width
	}
	if face > 1024 {
		face = 1024
	}

	// Adjust rect.
	SetAttrs(Obj(SplashRect), Attrs{
		"width": face, "height": face,
		"x": (ui.Width - face) / 2,
		"y": (ui.Height - face) / 2,
	})
	SetAttrs(Obj(SplashImage), Attrs{"width": face, "height": face})
}

func RemoveSplashScreen() {
	SplashRect.Remove()
	HasSplashScreen = false
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
	Canvas.Append(EndGameMessage)
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

// Alert window with given messate.
func Alert(msg string) {
	js.Global.Call("alert", msg)
}

// GetDocumentById from the DOM.
func GetDocumentById(id string) *js.Object {
	return document.Call("getElementById", id)
}

func (g *Game) OnCanvasResize() {
	ui.Width = Canvas.InnerWidth()
	ui.Height = Canvas.InnerHeight()

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
	AdjustOffBoardPieces()
	AdjustSplashScreen()
	g.AdjustEndGameMessagePosition()
	AdjustBusyBoxPosition()

	// Adjust all elements on page.
	OnChangeOfUIParams()
}

func OnChangeOfUIParams() {
	PiecesOnChangeOfUIParams()
	SelectionsOnChangeOfUIParams()
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
			RemovePiece(action)
		} else {
			RemoveOffBoardPiece(player, action)
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

	// Create UIParams.
	ui = NewUIParams()
	g.createBoardRects()
	CreateSplashScreen()
	g.PlaceOffBoardPieces()
	g.OnCanvasResize()

	Canvas.On("wheel", ZoomOnWheel)
	Canvas.On(jquery.MOUSEDOWN, DragOnMouseDown)
	Canvas.On(jquery.MOUSEUP, DragOnMouseUp)
	Canvas.On(jquery.MOUSEMOVE, DragOnMouseMove)
	jq(window).On(jquery.RESIZE, g.OnCanvasResize)

}
