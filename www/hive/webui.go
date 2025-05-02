package main

import (
	"fmt"
	"github.com/gowebapi/webapi/dom"
	"github.com/gowebapi/webapi/graphics/svg"
	"github.com/gowebapi/webapi/html"
	"github.com/gowebapi/webapi/html/htmlevent"
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
	splashRect      *svg.SVGRectElement
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
// The value will be converted to string using fmt.Sprintf("%s") -- except for numbers which will be converted
// using %d or %g depending on the underlying type.
type Attrs map[string]any

// CreateSVG returns a new untyped DOM element created with the SVG name space and the given elemType.
func CreateSVG(elemType string, attrs Attrs) *dom.Element {
	SVGNameSpace := "http://www.w3.org/2000/svg" // It's better to define it here or as a constant

	elem := Document.CreateElementNS(&SVGNameSpace, elemType, nil)
	SetAttrs(elem, attrs)
	return elem
}

// SetAttrs allow setting of various attributes at the same time.
func SetAttrs(elem *dom.Element, attrs Attrs) {
	for key, value := range attrs {
		switch v := value.(type) {
		case int:
			elem.SetAttribute(key, strconv.Itoa(v))
		case float32:
			elem.SetAttribute(key, strconv.FormatFloat(float64(v), 'f', -1, 32)) // 'f' for decimal, -1 for auto precision
		case float64:
			elem.SetAttribute(key, strconv.FormatFloat(v, 'f', -1, 64))
		default:
			elem.SetAttribute(key, fmt.Sprintf("%s", v)) // Fallback for other types (like strings)
		}
	}
}

// Window management --------------------------------------------------------------------------------------------------

// OnCanvasResize should be called when the main window is resized.
func (ui *WebUI) OnCanvasResize() {
	ui.Width = ui.canvas.ClientWidth()   // ?InnerWidth()
	ui.Height = ui.canvas.ClientHeight() // ?InnerHeight()
	ui.AdjustSplashScreen()

	/*
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
		g.AdjustEndGameMessagePosition()
		Old.AdjustBusyBoxPosition()

		// Adjust all elements on page.
		OnChangeOfUIParams()
	*/
}

// Splash Screen ------------------------------------------------------------------------------------------------------

// CreateSplashScreen and position it accordingly.
func (ui *WebUI) CreateSplashScreen() {
	ui.splashPattern = svg.SVGPatternElementFromJS(CreateSVG("pattern", Attrs{
		"id":           "splash",
		"patternUnits": "objectBoundingBox",
		"width":        1.0,
		"height":       1.0,
		"x":            -0.040,
		"y":            0,
	}).JSValue())
	ui.splashPattern.Style().SetProperty("background-color", "#B8BC33", nil)
	ui.splashImage = svg.SVGImageElementFromJS(CreateSVG("image", Attrs{
		"href":   "assets/Grasshopper.png",
		"width":  1024,
		"height": 1024,
	}).JSValue())
	ui.splashPattern.AppendChild(&ui.splashImage.Node)
	ui.defs.AppendChild(&ui.splashPattern.Node)

	ui.splashRect = svg.SVGRectElementFromJS(CreateSVG("rect", Attrs{
		"stroke":       "black",
		"stroke-width": 0,
		"border":       0,
		"padding":      0,
		"fill":         "url(#splash)",
	}).JSValue())
	ui.canvas.AppendChild(&ui.splashRect.Node)
	ui.splashRect.SetOnMouseUp(func(event *htmlevent.MouseEvent, currentTarget *svg.SVGElement) {
		ui.RemoveSplashScreen()
		//ui.OpenStartGameDialog()
	})
	ui.hasSplashScreen = true
	ui.AdjustSplashScreen()
}

// RemoveSplashScreen is called when someone clicks on the Splash screen.
func (ui *WebUI) RemoveSplashScreen() {
	ui.splashRect.Remove()
	ui.splashPattern.Remove()
	ui.hasSplashScreen = false
}

// AdjustSplashScreen at start time and at changes in size.
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
	SetAttrs(&ui.splashImage.Element, Attrs{"width": face, "height": face})
	SetAttrs(&ui.splashRect.Element, Attrs{
		"width": face, "height": face,
		"x": strconv.Itoa((ui.Width - face) / 2),
		"y": strconv.Itoa((ui.Height - face) / 2),
	})
}
