package main

import (
	"fmt"
	"log"
	"math"

	"github.com/gopherjs/gopherjs/js"
	"github.com/gopherjs/jquery"
	"github.com/janpfeifer/hiveGo/state"
)

//Convenience:
var jq = jquery.NewJQuery

const (
	INPUT  = "input#name"
	OUTPUT = "span#output"

	CANVAS_SVG = "svg#canvas"
	SVG_DEFS   = "defs#svgdefs"

	STANDARD_FACE = 33.0
)

var (
	window    = js.Global
	document  = js.Global.Get("document")
	canvas    = jq(CANVAS_SVG)
	canvasObj = Obj(canvas)
	svgDefs   = jq(SVG_DEFS)
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
		Width:      canvas.InnerWidth(),
		Height:     canvas.InnerHeight(),
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

func DragOnMouseUp(e jquery.Event) {
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
	MarkNextPlayer()
}

func MarkNextPlayer() {
	for ii := uint8(0); ii < state.NUM_PLAYERS; ii++ {
		width := 2.0 * ui.PixelRatio
		if board.NextPlayer == ii {
			width = 6 * ui.PixelRatio
		}
		SetAttrs(OffBoardRects[ii], Attrs{
			"stroke-width": width,
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
	MarkNextPlayer()
	AdjustOffBoardPieces()

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
	board *state.Board
)

func ExecuteAction(action state.Action) {
	board = board.Act(action)
	if board.IsFinished() {
		fmt.Printf("Animate end of game.")
		return
	}
	if len(board.Derived.Actions) == 0 {
		// Auto-execute skip action.
		if action.Piece == state.NO_PIECE {
			// Two skip actions in a row.
			log.Fatal("No moves avaialble to either players !?")
			return
		}
		// Recurse to a skip action.
		ExecuteAction(state.Action{Piece: state.NO_PIECE})
		return
	}

	// Select next player.
	MarkNextPlayer()
}

func main() {
	//show jQuery Version on console:
	print("Your current jQuery version is: " + jq().Jquery)

	// Create board parts.
	board = state.NewBoard()

	// Create UIParams.
	ui = NewUIParams()
	createBoardRects()
	PlaceOffBoardPieces(board)
	OnCanvasResize()

	canvas.On("wheel", ZoomOnWheel)
	canvas.On(jquery.MOUSEDOWN, DragOnMouseDown)
	canvas.On(jquery.MOUSEUP, DragOnMouseUp)
	canvas.On(jquery.MOUSEMOVE, DragOnMouseMove)
	jq(window).On(jquery.RESIZE, OnCanvasResize)

}

func ExamplePieces() {
	for ii := state.ANT; ii < state.LAST_PIECE_TYPE; ii++ {
		action := state.Action{
			Move:      false,
			Piece:     ii,
			TargetPos: state.Pos{int8(ii) - 3, 0}}
		Place(uint8(ii)%2, action)
	}
}
