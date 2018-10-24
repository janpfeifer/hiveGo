package main

import (
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
	canvasObj = canvas.Underlying().Index(0)
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
		PixelRatio: window.Get("devicePixelRatio").Float(),
		Width:      jq(window).InnerWidth(),
		Height:     jq(window).InnerHeight(),
		Scale:      1.0,
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

func Alert(msg string) {
	js.Global.Call("alert", msg)
}

func GetDocumentById(id string) *js.Object {
	return document.Call("getElementById", id)
}

func main() {

	//show jQuery Version on console:
	print("Your current jQuery version is: " + jq().Jquery)

	// Create UIParams.
	ui = NewUIParams()
	ui.Scale = 2.0
	OnChangeOfUIParams()

	canvas.On("wheel", ZoomOnWheel)
	canvas.On(jquery.MOUSEDOWN, DragOnMouseDown)
	canvas.On(jquery.MOUSEUP, DragOnMouseUp)
	canvas.On(jquery.MOUSEMOVE, DragOnMouseMove)

	for ii := state.ANT; ii < state.LAST_PIECE_TYPE; ii++ {
		action := state.Action{
			Move:      false,
			Piece:     ii,
			TargetPos: state.Pos{int8(ii) - 3, 0}}
		Place(int(ii)%2, action)
	}
}