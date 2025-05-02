package main

import (
	"github.com/gowebapi/webapi/dom"
	"github.com/gowebapi/webapi/graphics/svg"
	"github.com/gowebapi/webapi/html"
	"github.com/janpfeifer/hiveGo/internal/state"
	"strconv"
)

const (
	StandardFaceScale = 33.0
)

// WebUI implements the Hive Wasm UI
type WebUI struct {
	// HTML/SVG elements in the page
	canvas  *svg.SVGElement
	defs    *svg.SVGDefsElement
	busyBox *html.HTMLImageElement

	// Splash image:
	splashPattern   *svg.SVGPatternElement
	splashImage     *svg.SVGImageElement
	hasSplashScreen bool

	// PixelRatio gives a sense of how dense are pixels, where
	// 1.0 is "standard". It affects the zoom level of pieces off-g.board.
	PixelRatio float64

	// UI size: typically the Window size, but could be
	// larger, in case the Window is too little to fit everythin,
	// in which case things will be drawn off-screen.
	Width, Height int

	// Scale is controlled by the mouse wheel. It affects
	// only the board, not the pieces offg.board.
	Scale float64

	// How much the board has been dragged around.
	ShiftX, ShiftY float64
}

func NewWebUI() *WebUI {
	ui := &WebUI{
		// PixelRatio should be set to Window.Get("devicePixelRatio").Float(),
		// but at least on PixelBook, the browser already zoom things to
		// the PixelRatio without us needing to do anything. So fixed to 1.
		// for now.
		PixelRatio: 1.0,
		Scale:      Window.DevicePixelRatio(),
		ShiftX:     0,
		ShiftY:     0,
	}
	elem := Document.GetElementById("svg_canvas")
	ui.canvas = svg.SVGElementFromJS(elem.JSValue())
	elem = Document.GetElementById("svg_defs")
	ui.defs = svg.SVGDefsElementFromJS(elem.JSValue())
	elem = Document.GetElementById("busy")
	ui.busyBox = html.HTMLImageElementFromJS(elem.JSValue())
	ui.Width = ui.canvas.ClientWidth()   // InnerWidth ??
	ui.Height = ui.canvas.ClientHeight() // InnerHeight ??
	return ui
}

// SetBusy changes the "busy" status, displaying the animated gif.
func (ui *WebUI) SetBusy(busy bool) {
	if busy {
		ui.busyBox.Style().SetProperty("display", "block", nil)
	} else {
		ui.busyBox.Style().SetProperty("display", "none", nil)
	}
}

// Face returns the size of an hexagon's face.
func (ui *WebUI) Face() float64 {
	return StandardFaceScale * ui.Scale
}

// OffBoardHeight for UI.
func (ui *WebUI) OffBoardHeight() int {
	return int(128.0 * ui.PixelRatio)
}

// hexTriangleHeight returns the height of the triangles that make up for an hexagon, given the face lenght.
func hexTriangleHeight(face float64) float64 {
	return 0.866 * face // sqrt(3)/2 * face
}

// PosToXY converts board positions to screen position.
func (ui *WebUI) PosToXY(pos state.Pos, stackCount int) (x, y float64) {
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

var SVGNameSpace = "http://www.w3.org/2000/svg"

// Attrs are attributes to be set in a new example.
// Just a shortcut to a map[string]string.
type Attrs map[string]string

// CreateSVG returns a new untyped DOM element created with the SVG name space and the given elemType.
func CreateSVG(elemType string, attrs Attrs) *dom.Element {
	elem := Document.CreateElementNS(&SVGNameSpace, elemType, nil)
	for key, value := range attrs {
		elem.SetAttribute(key, value)
	}
	return elem
}

func (ui *WebUI) CreateSplashScreen() {
	ui.splashPattern = svg.SVGPatternElementFromJS(CreateSVG("pattern", Attrs{
		"id":           "splash",
		"patternUnits": "objectBoundingBox",
		"width":        "1.0",
		"height":       "1.0",
		"x":            "-0.040",
		"y":            "0",
	}).JSValue())
	ui.splashImage = svg.SVGImageElementFromJS(CreateSVG("image", Attrs{
		"href":   "assets/Grasshopper.png",
		"width":  "1024",
		"height": "1024",
	}).JSValue())
	ui.splashPattern.AppendChild(&ui.splashImage.Node)
	ui.defs.AppendChild(&ui.splashPattern.Node)

	/*
		SplashRect = jq(CreateSVG("rect", Attrs{
			"stroke":       "black",
			"stroke-width": 0,
			"border":       0,
			"padding":      0,
			"fill":         "url(#splash)",
		}))
		canvas.Append(SplashRect)
		SplashRect.On(jquery.MOUSEUP, func(e jquery.Event) {
			RemoveSplashScreen()
			OpenStartGameDialog()
		})
		AdjustSplashScreen()
	*/
	ui.hasSplashScreen = true
	ui.AdjustSplashScreen()
}

func (ui *WebUI) AdjustSplashScreen() {
	if !ui.hasSplashScreen {
		return
	}

	// SplashScreen will have at most 1024 "face" size (width = height = face).
	face := ui.Height - 2*ui.OffBoardHeight()
	if ui.Width < face {
		face = ui.Width
	}
	if face > 1024 {
		face = 1024
	}

	// Adjust rect.
	/*
		SetAttrs(Obj(SplashRect), Attrs{
			"width": face, "height": face,
			"x": (ui.Width - face) / 2,
			"y": (ui.Height - face) / 2,
		})
	*/
	faceStr := strconv.Itoa(face)
	ui.splashImage.SetAttribute("width", faceStr)
	ui.splashImage.SetAttribute("height", faceStr)
}
