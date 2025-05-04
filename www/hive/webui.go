package main

import (
	"fmt"
	"github.com/gowebapi/webapi/dom"
	"github.com/gowebapi/webapi/dom/domcore"
	"github.com/gowebapi/webapi/graphics/svg"
	"github.com/gowebapi/webapi/html"
	"github.com/gowebapi/webapi/html/htmlevent"
	"github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"strconv"
)

// ==================================================================================================================
// WebUI ------------------------------------------------------------------------------------------------------------
// ==================================================================================================================

// WebUI implements the Hive Wasm UI
type WebUI struct {
	board *state.Board

	// HTML/SVG elements in the page
	canvas  *svg.SVGElement
	defs    *svg.SVGDefsElement
	busyBox *html.HTMLImageElement

	// Splash image:
	splashDiv       *html.HTMLDivElement
	hasSplashScreen bool

	// Game start dialog:
	gameStartDialog     *html.HTMLDivElement
	gameStartAIConfig   *html.HTMLInputElement
	aiConfig            string
	isHotseat, aiStarts bool

	// Board UI elements:
	boardGroup     *svg.SVGGElement
	offBoardGroups [2]*svg.SVGGElement
	offBoardRects  [2]*svg.SVGRectElement

	// Elements for pieces on the board:
	onBoardPiecesPatterns, offBoardPiecesPatterns []*svg.SVGPatternElement
	onBoardPiecesImages, offBoardPiecesImages     []*svg.SVGImageElement
	onBoardTilesPatterns, offBoardTilesPatterns   [state.NumPlayers]*svg.SVGPatternElement
	onBoardTilesImages, offBoardTilesImages       [state.NumPlayers]*svg.SVGImageElement

	// Map of all pieces currently on board:
	piecesOnBoard      map[state.Pos][]*PieceOnScreen
	piecesOffBoard     [state.NumPlayers]map[state.PieceType][]*PieceOnScreen
	piecesOnBoardIndex int

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
		// PixelRatio should be set to Window.Get("devicePixelRatio").Float().
		PixelRatio: Window.DevicePixelRatio(),
		Scale:      1.0,
		ShiftX:     0,
		ShiftY:     0,

		piecesOnBoard: make(map[state.Pos][]*PieceOnScreen),
	}
	elem := Document.GetElementById("svg_canvas")
	ui.canvas = svg.SVGElementFromJS(elem.JSValue())
	elem = Document.GetElementById("svg_defs")
	ui.defs = svg.SVGDefsElementFromJS(elem.JSValue())
	elem = Document.GetElementById("busy")
	ui.busyBox = html.HTMLImageElementFromJS(elem.JSValue())
	ui.Width = ui.canvas.ClientWidth()   // InnerWidth ??
	ui.Height = ui.canvas.ClientHeight() // InnerHeight ??

	// Board area:
	ui.createBoardRects()

	// Create patterns for pieces and its tiles:
	ui.onBoardPiecesPatterns, ui.onBoardPiecesImages =
		ui.createPiecesPatternsAndImages(OnBoard)
	ui.offBoardPiecesPatterns, ui.offBoardPiecesImages =
		ui.createPiecesPatternsAndImages(OffBoard)
	for playerNum := range state.PlayerInvalid {
		elem := Document.GetElementById(playerToTilePatternID(OffBoard, playerNum))
		if elem == nil {
			klog.Fatalf("Failed to find element %q", playerToTilePatternID(OffBoard, playerNum))
		}
		ui.offBoardTilesPatterns[playerNum] = svg.SVGPatternElementFromWrapper(elem)
		ui.offBoardTilesImages[playerNum] = svg.SVGImageElementFromWrapper(ui.offBoardTilesPatterns[playerNum].FirstElementChild())
		ui.onBoardTilesPatterns[playerNum] = svg.SVGPatternElementFromWrapper(Document.GetElementById(playerToTilePatternID(OnBoard, playerNum)))
		ui.onBoardTilesImages[playerNum] = svg.SVGImageElementFromWrapper(ui.onBoardTilesPatterns[playerNum].FirstElementChild())
	}
	return ui
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

// ==================================================================================================================
// Window management ------------------------------------------------------------------------------------------------
// ==================================================================================================================

// UpdateBoard will update the display of status to the current board:
// Current player, etc.
func (ui *WebUI) UpdateBoard(board *state.Board) {
	ui.board = board
	ui.MarkNextPlayer()
}

// OnCanvasResize should be called when the main window is resized.
func (ui *WebUI) OnCanvasResize() {
	ui.Width = ui.canvas.ClientWidth()   // ?InnerWidth()
	ui.Height = ui.canvas.ClientHeight() // ?InnerHeight()

	// OffBoard space.
	offboardHeight := ui.OffBoardHeight()
	if ui.Height < 3*offboardHeight {
		ui.Height = 3 * offboardHeight
	}
	SetAttrs(&ui.offBoardRects[0].Element, Attrs{
		"x": 0, "y": 0,
		"width":  ui.Width,
		"height": offboardHeight,
	})
	SetAttrs(&ui.offBoardRects[1].Element, Attrs{
		"x": 0, "y": ui.Height - offboardHeight,
		"width":  ui.Width,
		"height": offboardHeight,
	})
	ui.AdjustOffBoardPieces()
	ui.AdjustOnBoardPieces()
	ui.MarkNextPlayer()

	/*
		g.AdjustEndGameMessagePosition()
		Old.AdjustBusyBoxPosition()

		// Adjust all elements on page.
		OnChangeOfUIParams()
	*/
}

// SetBusy changes the "busy" status, displaying the animated gif.
func (ui *WebUI) SetBusy(busy bool) {
	if busy {
		ui.busyBox.Style().SetProperty("display", "block", nil)
	} else {
		ui.busyBox.Style().SetProperty("display", "none", nil)
	}
}

// ==================================================================================================================
// Splash Screen ----------------------------------------------------------------------------------------------------
// ==================================================================================================================

// CreateSplashScreen and position it accordingly.
// onClose is called once the splash screen is closed.
func (ui *WebUI) CreateSplashScreen(onClose func()) {
	if ui.splashDiv == nil {
		elem := Document.GetElementById("splashScreen")
		ui.splashDiv = html.HTMLDivElementFromWrapper(elem)
	}
	ui.splashDiv.Style().SetProperty("display", "flex", nil)
	doneFn := func() {
		fmt.Println("Closing splash screen")
		ui.RemoveSplashScreen()
		onClose()
	}
	ui.splashDiv.AddEventListener("click", domcore.NewEventListenerFunc(func(event *domcore.Event) {
		doneFn()
	}), nil)
	ui.splashDiv.AddEventListener("keyup", domcore.NewEventListenerFunc(func(event *domcore.Event) {
		event.PreventDefault()
		doneFn()
	}), nil)
	ui.splashDiv.Focus(nil)
	ui.hasSplashScreen = true
}

// RemoveSplashScreen is called when someone clicks on the Splash screen.
func (ui *WebUI) RemoveSplashScreen() {
	ui.splashDiv.Style().SetProperty("display", "none", nil)
	ui.hasSplashScreen = false
}

// ==================================================================================================================
// Game Dialog ------------------------------------------------------------------------------------------------------
// ==================================================================================================================

var levelsConfigs = map[string]string{
	"easy":   "fnn=#0;ab;max_depth=2;randomness=0.1",              // Easy
	"medium": "a0fnn=#0;mcts;max_time=2s;temperature=1.5",         // Medium
	"hard":   "a0fnn=#0;mcts;max_traverses=10000;temperature=0.1", // Hard
}

// OpenGameStartDialog and manages the game dialog.
func (ui *WebUI) OpenGameStartDialog(onStart func()) {
	if ui.gameStartDialog == nil {
		ui.gameStartDialog = html.HTMLDivElementFromWrapper(Document.GetElementById("new_game"))
		ui.gameStartAIConfig = html.HTMLInputElementFromWrapper(ui.gameStartDialog.QuerySelector("input#ai_config"))
	}

	ui.gameStartAIConfig.SetValue(levelsConfigs["easy"])
	for key, value := range levelsConfigs {
		radio := html.HTMLInputElementFromWrapper(ui.gameStartDialog.QuerySelector("input#" + key))
		radio.SetOnClick(func(event *htmlevent.MouseEvent, currentTarget *html.HTMLElement) {
			ui.gameStartAIConfig.SetValue(value)
		})
	}
	ui.gameStartDialog.Style().SetProperty("display", "flex", nil)

	form := html.HTMLFormElementFromWrapper(ui.gameStartDialog.QuerySelector("form"))
	form.SetOnSubmit(func(event *domcore.Event, currentTarget *html.HTMLElement) {
		// Collect inputs:
		ui.aiConfig = ui.gameStartAIConfig.Value()

		event.PreventDefault()
		elem := ui.gameStartDialog.QuerySelector("#hotseat")
		if elem == nil {
			klog.Fatal("Failed to find radio button for hotseat")
		}
		hotseat := html.HTMLInputElementFromWrapper(elem)
		ui.isHotseat = hotseat.Checked()

		elem = ui.gameStartDialog.QuerySelector("#ai_starts")
		if elem == nil {
			klog.Fatal("Failed to find radio button for ai_starts checkbox")
		}
		ui.aiStarts = html.HTMLInputElementFromWrapper(elem).Checked()

		// Disable dialog.
		ui.gameStartDialog.Style().SetProperty("display", "none", nil)

		// Start caller's onStart.
		onStart()
	})

	// Focus the first input element of the form.
	elem := form.QuerySelector("input:not([disabled])")
	html.HTMLInputElementFromWrapper(elem).Focus(nil)
}

// AIConfig returns the AI configuration string read from the dialog form.
func (ui *WebUI) AIConfig() string {
	return ui.aiConfig
}

func (ui *WebUI) IsHotseat() bool {
	return ui.isHotseat
}

func (ui *WebUI) AIStarts() bool { return ui.aiStarts }

// ==================================================================================================================
// Board UI ---------------------------------------------------------------------------------------------------------
// ==================================================================================================================

// StartBoard (the UI) with the given board (the state).
// It initializes the off-board
func (ui *WebUI) StartBoard(board *state.Board) {
	ui.board = board
	ui.canvas.Style().SetProperty("display", "block", nil)
	ui.canvas.Style().SetProperty("pointer-events", "all", nil)
	ui.CreateOffBoardPieces()
	ui.OnCanvasResize()
}

func (ui *WebUI) createBoardRects() {
	ui.boardGroup = svg.SVGGElementFromWrapper(CreateSVG("g", Attrs{
		"x":      0,
		"y":      0,
		"width":  "100%",
		"height": "100%",
	}))
	ui.canvas.AppendChild(&ui.boardGroup.Node)
	for ii := range 2 {
		ui.offBoardGroups[ii] = svg.SVGGElementFromWrapper(CreateSVG("g", Attrs{
			"x":      0,
			"y":      0,
			"width":  "100%",
			"height": "100%",
		}))
		ui.canvas.AppendChild(&ui.offBoardGroups[ii].Node)
		ui.offBoardRects[ii] = svg.SVGRectElementFromWrapper(CreateSVG("rect", Attrs{
			"stroke": "firebrick",
			"fill":   "moccasin",
		}))
		ui.offBoardGroups[ii].AppendChild(&ui.offBoardRects[ii].Node)
	}
}

// MarkNextPlayer to play. It also highlights the winner if the game is finished.
func (ui *WebUI) MarkNextPlayer() {
	if ui.board == nil {
		// No board set:
		return
	}
	for player := range state.PlayerNum(state.NumPlayers) {
		width := 1.0 * ui.PixelRatio
		stroke := "firebrick"
		if ui.board.IsFinished() {
			if ui.board.Draw() || ui.board.Winner() == player {
				stroke = "url(#colors)"
				width = 6 * ui.PixelRatio
			}
		} else {
			if ui.board.NextPlayer == player {
				width = 3 * ui.PixelRatio
			}
		}
		SetAttrs(&ui.offBoardRects[player].Element, Attrs{
			"stroke-width": width,
			"stroke":       stroke,
		})
	}
}
