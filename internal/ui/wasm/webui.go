package main

import (
	"fmt"
	state2 "github.com/janpfeifer/hiveGo/internal/state"
	"log"
	"math"

	"github.com/gopherjs/gopherjs/js"
	"github.com/gopherjs/jquery"
)

// Convenience:
var jq = jquery.NewJQuery

const (
	CANVAS_SVG = "svg#canvas"
	SVG_DEFS   = "defs#svgdefs"

	STANDARD_FACE = 33.0
)

var (
	window   = js.Global
	document = js.Global.Get("document")
	Canvas   = jq(CANVAS_SVG)
	SvgDefs  = jq(SVG_DEFS)
)

type UIParams struct {
	// PixelRatio gives a sense of how dense are pixels, where
	// 1.0 is "standard". It affects the zoom level of pieces off-board.
	PixelRatio float64

	// UI size: typically the window size, but could be
	// larger, in case the window is too little to fit everythin,
	// in which case things will be drawn off-screen.
	Width, Height int

	// Scale is controlled by the mouse wheel. It affects
	// only the board, not the pieces offboard.
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
	return STANDARD_FACE * ui.Scale
}

// OffBoard height for UI.
func (ui *UIParams) OffBoardHeight() int {
	return int(128.0 * ui.PixelRatio)
}

// hexTriangleHeight returns the height of the triangles that make up for an hexagon, given the face lenght.
func hexTriangleHeight(face float64) float64 {
	return 0.866 * face // sqrt(3)/2 * face
}

// Convert board positions to screen position.
func (ui *UIParams) PosToXY(pos state2.Pos, stackCount int) (x, y float64) {
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

// Control Zooming of board.
func ZoomOnWheel(e jquery.Event) {
	wheelEvent := e.Object.Get("originalEvent")
	scrollAmount := wheelEvent.Get("deltaY").Float()
	ui.Scale = ui.Scale * math.Pow(1.1, scrollAmount/50.0)
	OnChangeOfUIParams()
}

// Variables that control the drag of the board.
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

func createBoardRects() {
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
	MarkNextPlayer()
}

func MarkNextPlayer() {
	for ii := uint8(0); ii < state2.NumPlayers; ii++ {
		width := 2.0 * ui.PixelRatio
		stroke := "firebrick"
		if Board.IsFinished() {
			if Board.Draw() || Board.Winner() == ii {
				stroke = "url(#colors)"
				width = 12 * ui.PixelRatio
			}
		} else {
			if Board.NextPlayer == ii {
				width = 6 * ui.PixelRatio
			}
		}
		SetAttrs(OffBoardRects[ii], Attrs{
			"stroke-width": width,
			"stroke":       stroke,
		})
	}
}

var (
	// Reference to splash screen objects. Created at the beginning.
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
	Board = state2.NewBoard()
	IsRunning = true

	// Prepare game.
	gameType := jq("input[name=game_type]:checked").Val()
	StartGameDialog.Div.SetCss("display", "none")
	if gameType == "ai" {
		aiConfig := jq("input[name=ai_config]").Val()
		fmt.Printf("AI: config=%s, starts=%s\n", aiConfig,
			jq("input[name=ai_starts]:checked").Val())
		aiPlayer := 1
		if jq("input[name=ai_starts]:checked").Val() == "on" {
			aiPlayer = 0
		}
		StartAI(aiConfig, aiPlayer)
	}
}

var (
	HasEndGameMessage = false
	EndGameMessage    jquery.JQuery
)

func ShowEndGameMessage(text string) {
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
	AdjustEndGameMessagePosition()
}

func AdjustEndGameMessagePosition() {
	if HasEndGameMessage {
		var y float64
		if Board.Draw() {
			y = float64(ui.Height) / 2.0
		} else if Board.Winner() == 0 {
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

// Tools
func Alert(msg string) {
	js.Global.Call("alert", msg)
}

func GetDocumentById(id string) *js.Object {
	return document.Call("getElementById", id)
}

func OnCanvasResize() {
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
	MarkNextPlayer()
	AdjustOffBoardPieces()
	AdjustSplashScreen()
	AdjustEndGameMessagePosition()
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

// Game information.
var (
	Board     *state2.Board
	IsRunning = false
)

func ExecuteAction(action state2.Action) {
	// Animate action.
	if action.Piece != state2.NoPiece {
		player := Board.NextPlayer
		if action.Move {
			RemovePiece(action)
		} else {
			RemoveOffBoardPiece(player, action)
		}
		Place(player, action)
	}

	Board = Board.Act(action)
	if Board.IsFinished() {
		MarkNextPlayer()
		var msg string
		if Board.Draw() {
			msg = "Draw !!!"
		} else if Board.Winner() == 0 {
			msg = "Top Player Wins !!!"
		} else {
			msg = "Bottom Player Wins !!!"
		}
		ShowEndGameMessage(msg)
		IsRunning = false
		return
	}

	if len(Board.Derived.Actions) == 0 {
		// Auto-execute skip action.
		if action.Piece == state2.NoPiece {
			// Two skip actions in a row.
			log.Fatal("No moves avaialble to either players !?")
			return
		}
		// Recurse to a skip action.
		ExecuteAction(state2.Action{Piece: state2.NoPiece})
		return
	}

	// Select next player.
	MarkNextPlayer()
	ScheduleAIPlay()
}

func main() {
	//show jQuery Version on console:
	print("Your current jQuery version is: " + jq().Jquery)

	// Create board parts.
	Board = state2.NewBoard()
	IsRunning = false

	// Create UIParams.
	ui = NewUIParams()
	createBoardRects()
	CreateSplashScreen()
	PlaceOffBoardPieces(Board)
	OnCanvasResize()

	Canvas.On("wheel", ZoomOnWheel)
	Canvas.On(jquery.MOUSEDOWN, DragOnMouseDown)
	Canvas.On(jquery.MOUSEUP, DragOnMouseUp)
	Canvas.On(jquery.MOUSEMOVE, DragOnMouseMove)
	jq(window).On(jquery.RESIZE, OnCanvasResize)

}
