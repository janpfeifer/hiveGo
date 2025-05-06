package main

import (
	"fmt"
	"github.com/gowebapi/webapi"
	"github.com/gowebapi/webapi/core/js"
	"github.com/gowebapi/webapi/dom"
	"github.com/gowebapi/webapi/dom/domcore"
	"github.com/gowebapi/webapi/graphics/svg"
	"github.com/gowebapi/webapi/html"
	"github.com/gowebapi/webapi/html/htmlevent"
	"github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
	"math"
	"strconv"
	"time"
)

// ==================================================================================================================
// WebUI ------------------------------------------------------------------------------------------------------------
// ==================================================================================================================

type UIState int

const (
	StateSplashScreen UIState = iota
	StateConfig
	StateGame
)

// WebUI implements the Hive Wasm UI
type WebUI struct {
	state UIState

	// board is the current board being displayed if in game.
	board *state.Board

	// HTML/SVG elements in the page
	canvas *svg.SVGElement
	defs   *svg.SVGDefsElement

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
	piecesOnBoard    map[state.Pos][]*PieceOnScreen
	piecesOffBoard   [state.NumPlayers]map[state.PieceType][]*PieceOnScreen
	piecesOnBoardIdx int

	// selections: used to show the user the valid move options.
	selections Selections

	// Tutorial objects:
	isTutorialOn                 bool
	tutorialCloseButton          *html.HTMLButtonElement
	tutorialContent, tutorialBox *html.HTMLDivElement
	tutorialTitle                *html.HTMLSpanElement

	// Game over message box:
	gameOverBox   *html.HTMLDivElement
	winnerMessage *html.HTMLParagraphElement
	restartButton *html.HTMLButtonElement

	// Status Box: time and AI evals/second:
	statusBox         *html.HTMLDivElement
	playersClocks     [2]*html.HTMLSpanElement
	aiEvalRateDiv     *html.HTMLDivElement
	aiEvalRateSpan    *html.HTMLSpanElement
	lastAccountedTime time.Time
	playersTimes      [2]time.Duration
	isClockRunning    bool

	// UI Buttons
	uiButtonsBox *html.HTMLDivElement

	// Help page
	isHelpOpen bool
	helpPage   *html.HTMLDivElement

	// PixelRatio is a characteristic of the user's display: it gives a sense of how dense are pixels, where
	// 1.0 is "standard". It affects the scaling of the "standard" size (off-board pieces, and original on-board pieces).
	// Notice it changes if the user changes the zoom level using the browser (in Chrome using control+plus and control-minus).
	PixelRatio float64

	// Scale is controlled by the mouse wheel. It affects
	// only the board, not the pieces off-board.
	Scale float64

	// UI size: typically the Window size, but could be
	// larger, in case the Window is too little to fit everything,
	// in which case things will be drawn off-screen.
	Width, Height int

	// How much the board has been dragged around.
	ShiftX, ShiftY float64

	// Dragging event:
	dragStarted  bool
	dragX, dragY float64
}

func NewWebUI() *WebUI {
	ui := &WebUI{
		// PixelRatio should be set to Window.Get("devicePixelRatio").Float().
		PixelRatio:    Window.DevicePixelRatio(),
		Scale:         1.0,
		ShiftX:        0,
		ShiftY:        0,
		piecesOnBoard: make(map[state.Pos][]*PieceOnScreen),
		isTutorialOn:  true,
	}

	// Canvas:
	elem := Document.GetElementById("svg_canvas")
	ui.canvas = svg.SVGElementFromJS(elem.JSValue())
	elem = Document.GetElementById("svg_defs")
	ui.defs = svg.SVGDefsElementFromJS(elem.JSValue())
	ui.Width = ui.canvas.ClientWidth()
	ui.Height = ui.canvas.ClientHeight()

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

	// Tutorial box:
	elem = Document.GetElementById("tutorialBox")
	ui.tutorialBox = html.HTMLDivElementFromWrapper(elem)
	elem = Document.GetElementById("tutorialClose")
	ui.tutorialCloseButton = html.HTMLButtonElementFromWrapper(elem)
	ui.tutorialCloseButton.SetOnClick(func(event *htmlevent.MouseEvent, currentTarget *html.HTMLElement) { ui.CloseTutorial() })
	elem = Document.GetElementById("tutorialContent")
	ui.tutorialContent = html.HTMLDivElementFromWrapper(elem)
	elem = Document.GetElementById("tutorialTitle")
	ui.tutorialTitle = html.HTMLSpanElementFromWrapper(elem)

	// Game Over message box:
	elem = Document.GetElementById("gameOverBox")
	ui.gameOverBox = html.HTMLDivElementFromWrapper(elem)
	elem = Document.GetElementById("winnerMessage")
	ui.winnerMessage = html.HTMLParagraphElementFromWrapper(elem)
	elem = Document.GetElementById("restartGame")
	ui.restartButton = html.HTMLButtonElementFromWrapper(elem)
	ui.restartButton.SetOnClick(func(event *htmlevent.MouseEvent, currentTarget *html.HTMLElement) {
		Window.Location().Reload()
	})

	// Status box:
	elem = Document.GetElementById("statusBox")
	ui.statusBox = html.HTMLDivElementFromWrapper(elem)
	ui.playersClocks[0] = html.HTMLSpanElementFromWrapper(Document.GetElementById("player0Clock"))
	ui.playersClocks[1] = html.HTMLSpanElementFromWrapper(Document.GetElementById("player1Clock"))
	ui.aiEvalRateDiv = html.HTMLDivElementFromWrapper(Document.GetElementById("aiEvalRate"))
	ui.aiEvalRateSpan = html.HTMLSpanElementFromWrapper(Document.GetElementById("evalsPerSec"))

	// Help page
	elem = Document.GetElementById("help-page")
	ui.helpPage = html.HTMLDivElementFromWrapper(elem)
	ui.helpPage.SetOnKeyUp(func(event *htmlevent.KeyboardEvent, currentTarget *html.HTMLElement) {
		fmt.Printf("Key %q pressed in help page\n", event.Key())
		if event.Key() == "Escape" {
			ui.CloseHelp()
		}
		event.PreventDefault()
	})
	closeHelpBtn := html.HTMLButtonElementFromWrapper(ui.helpPage.QuerySelector("button"))
	closeHelpBtn.SetOnClick(func(_ *htmlevent.MouseEvent, _ *html.HTMLElement) {
		ui.CloseHelp()
	})

	// UI-buttons
	elem = Document.GetElementById("ui-buttons")
	ui.uiButtonsBox = html.HTMLDivElementFromWrapper(elem)
	buttonHome := html.HTMLButtonElementFromWrapper(ui.uiButtonsBox.QuerySelector("#btn-home"))
	buttonHome.SetOnClick(func(_ *htmlevent.MouseEvent, _ *html.HTMLElement) {
		ui.Home()
	})
	buttonZoomIn := html.HTMLButtonElementFromWrapper(ui.uiButtonsBox.QuerySelector("#btn-zoom-in"))
	buttonZoomIn.SetOnClick(func(_ *htmlevent.MouseEvent, _ *html.HTMLElement) {
		ui.Scale *= 1.5
		ui.OnCanvasResize()
	})
	buttonZoomOut := html.HTMLButtonElementFromWrapper(ui.uiButtonsBox.QuerySelector("#btn-zoom-out"))
	buttonZoomOut.SetOnClick(func(_ *htmlevent.MouseEvent, _ *html.HTMLElement) {
		ui.Scale *= 1.0 / 1.5
		ui.OnCanvasResize()
	})
	buttonHelp := html.HTMLButtonElementFromWrapper(ui.uiButtonsBox.QuerySelector("#btn-help"))
	buttonHelp.SetOnClick(func(_ *htmlevent.MouseEvent, _ *html.HTMLElement) {
		if ui.isHelpOpen {
			ui.CloseHelp()
		} else {
			ui.OpenHelp()
		}
	})

	ui.selectionsInit() // Actions selection mechanism.
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
// Setting the value to nil is equivalent to removing it.
func SetAttrs(elem *dom.Element, attrs Attrs) {
	for key, value := range attrs {
		if value == nil {
			elem.RemoveAttribute(key)
			continue
		}
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
	ui.AdjustSelections()

	// Adjust ui-buttons position.
	ui.uiButtonsBox.Style().SetProperty("top", fmt.Sprintf("%dpx", offboardHeight+20), nil)

	/*
		g.AdjustEndGameMessagePosition()
		Old.AdjustBusyBoxPosition()

		// Adjust all elements on page.
		OnChangeOfUIParams()
	*/
}

// ==================================================================================================================
// Splash Screen ----------------------------------------------------------------------------------------------------
// ==================================================================================================================

// CreateSplashScreen and position it accordingly.
// onClose is called once the splash screen is closed.
func (ui *WebUI) CreateSplashScreen(onClose func()) {
	ui.state = StateSplashScreen

	if ui.splashDiv == nil {
		elem := Document.GetElementById("splashScreen")
		ui.splashDiv = html.HTMLDivElementFromWrapper(elem)
	}
	ui.splashDiv.Style().SetProperty("display", "flex", nil)
	doneFn := func() {
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
	"easy":   "fnn=#0,ab,max_depth=1,randomness=0.2",              // Easy
	"medium": "fnn=#0,ab,max_depth=2,randomness=0.05",             // Medium
	"hard":   "a0fnn=#0,mcts,max_traverses=10000,temperature=0.1", // Hard
}

// OpenGameStartDialog and manages the game dialog.
func (ui *WebUI) OpenGameStartDialog(onStart func()) {
	ui.state = StateConfig

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
	ui.state = StateGame

	ui.board = board
	ui.canvas.Style().SetProperty("display", "block", nil)
	ui.canvas.Style().SetProperty("pointer-events", "all", nil)
	ui.CreateOffBoardPieces()
	ui.ShowTutorial()
	ui.OnCanvasResize()

	ui.canvas.SetOnWheel(func(event *htmlevent.WheelEvent, currentTarget *svg.SVGElement) {
		ui.ZoomOnWheel(event)
	})
	ui.canvas.SetOnMouseDown(func(event *htmlevent.MouseEvent, currentTarget *svg.SVGElement) {
		ui.DragOnMouseDown(event)
	})
	ui.canvas.SetOnMouseUp(func(event *htmlevent.MouseEvent, currentTarget *svg.SVGElement) {
		ui.DragOnMouseUp(event)
	})
	ui.canvas.SetOnMouseMove(func(event *htmlevent.MouseEvent, currentTarget *svg.SVGElement) {
		ui.DragOnMouseMove(event)
	})
	ui.canvas.SetOnKeyDown(func(event *htmlevent.KeyboardEvent, currentTarget *svg.SVGElement) {
		if event.Key() == "F1" {
			event.PreventDefault()
			ui.OnKeyPress(event)
		}
	})
	ui.canvas.SetOnKeyUp(func(event *htmlevent.KeyboardEvent, currentTarget *svg.SVGElement) {
		ui.OnKeyPress(event)
	})
	//canvas.On(jquery.MOUSEUP, DragOnMouseUp)
	//canvas.On(jquery.MOUSEMOVE, DragOnMouseMove)
	Window.SetOnResize(func(_ *htmlevent.UIEvent, _ *webapi.Window) {
		ui.OnCanvasResize()
	})
	ui.canvas.Focus(nil)

	// Start other floating boxes:
	ui.StartStatusBox()
	ui.uiButtonsBox.Style().SetProperty("display", "flex", nil)
}

func (ui *WebUI) OnKeyPress(event *htmlevent.KeyboardEvent) {
	fmt.Printf("Key %q pressed in canvas\n", event.Key())
	if event.Key() == "Escape" && ui.selections.isSelecting {
		ui.cancelSelection()
		event.PreventDefault()
		return
	}
	if event.Key() == "Home" {
		ui.Home()
		event.PreventDefault()
		return
	}
	if event.Key() == "F1" {
		ui.OpenHelp()
		event.PreventDefault()
		return
	}
	return
}

// Home centers the board and set the zoom levels back to the default.
func (ui *WebUI) Home() {
	ui.Scale = 1.0
	ui.ShiftX = 0
	ui.ShiftY = 0
	ui.OnCanvasResize()
}

// ZoomOnWheel responds to mouse wheel movement to control zoom.
func (ui *WebUI) ZoomOnWheel(event *htmlevent.WheelEvent) {
	scrollAmount := event.DeltaY()
	ui.Scale = ui.Scale * math.Pow(1.1, -scrollAmount/50.0)
	//fmt.Printf("Wheel event: new scale is %g\n", ui.Scale)
	ui.OnCanvasResize()
}

// DragOnMouseDown starts a drag board event.
func (ui *WebUI) DragOnMouseDown(event *htmlevent.MouseEvent) {
	ui.dragStarted = true
	ui.dragX, ui.dragY = event.PageX(), event.PageY()
}

// DragOnMouseUp release the drag event.
func (ui *WebUI) DragOnMouseUp(event *htmlevent.MouseEvent) {
	ui.dragStarted = false
}

// DragOnMouseMove release the drag event.
func (ui *WebUI) DragOnMouseMove(event *htmlevent.MouseEvent) {
	if !ui.dragStarted {
		return
	}
	mouseX, mouseY := event.PageX(), event.PageY()
	deltaX, deltaY := mouseX-ui.dragX, mouseY-ui.dragY
	ui.ShiftX += float64(deltaX)
	ui.ShiftY += float64(deltaY)
	ui.dragX, ui.dragY = mouseX, mouseY

	// We only need to repaint the on-board area, the off-board area remains untouched.
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
			"fill":   "#AA5",
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

// ==================================================================================================================
// Tutorial ---------------------------------------------------------------------------------------------------------
// ==================================================================================================================

func (ui *WebUI) ShowTutorial() {
	ui.tutorialBox.Style().SetProperty("display", "block", nil)
	if ui.board != nil {
		style := ui.tutorialBox.Style()
		offBoardHeight := strconv.Itoa(ui.OffBoardHeight()+20) + "px"
		if ui.board.NextPlayer == state.PlayerFirst {
			style.SetProperty("top", offBoardHeight, nil)
			style.RemoveProperty("bottom")
		} else {
			style.SetProperty("bottom", offBoardHeight, nil)
			style.RemoveProperty("top")
		}
	}
}

func (ui *WebUI) HideTutorial() {
	ui.tutorialBox.Style().SetProperty("display", "none", nil)
}

func (ui *WebUI) SetTutorialTitle(title string) {
	ui.tutorialTitle.SetInnerHTML(title)
}

func (ui *WebUI) SetTutorialContent(content string) {
	ui.tutorialContent.SetInnerHTML(content)
}

func (ui *WebUI) CloseTutorial() {
	ui.isTutorialOn = false
	ui.HideTutorial()
}

// ==================================================================================================================
// Game Over --------------------------------------------------------------------------------------------------------
// ==================================================================================================================

func (ui *WebUI) SetWinner(message string) {
	ui.gameOverBox.Style().SetProperty("display", "block", nil)
	ui.winnerMessage.SetInnerHTML(message)
}

// ==================================================================================================================
// StatusBox --------------------------------------------------------------------------------------------------------
// ==================================================================================================================

var clockUpdateMilliseconds = int(1000)

func (ui *WebUI) StartStatusBox() {
	ui.statusBox.Style().SetProperty("display", "block", nil)
	ui.lastAccountedTime = time.Now()
	ui.isClockRunning = true
	clockCallbackFunc := js.FuncOf(func(js.Value, []js.Value) interface{} {
		if !ui.isClockRunning {
			return nil
		}
		ui.UpdateTime()
		return nil
	})
	Window.SetInterval(webapi.UnionFromJS(clockCallbackFunc.Value), &clockUpdateMilliseconds)
}

func (ui *WebUI) StopClocks() {
	ui.isClockRunning = false
}

func (ui *WebUI) UpdateAIEvalRate(aiEvalRate float64) {
	ui.aiEvalRateDiv.Style().SetProperty("display", "block", nil)
	ui.aiEvalRateSpan.SetInnerHTML(fmt.Sprintf("%.1f", aiEvalRate))
}

func (ui *WebUI) UpdateTime() {
	if !ui.isClockRunning {
		return
	}
	now := time.Now()
	elapsed := now.Sub(ui.lastAccountedTime)
	ui.lastAccountedTime = now
	playerNum := ui.board.NextPlayer
	ui.playersTimes[playerNum] += elapsed

	// Generate display time:
	d := ui.playersTimes[playerNum]
	hours := d / time.Hour
	d -= hours * time.Hour
	minutes := d / time.Minute
	d -= minutes * time.Minute
	seconds := d / time.Second
	if hours > 0 {
		ui.playersClocks[playerNum].SetInnerHTML(fmt.Sprintf("%d:%02d:%02d", hours, minutes, seconds))
	} else {
		ui.playersClocks[playerNum].SetInnerHTML(fmt.Sprintf("%02d:%02d", minutes, seconds))
	}
}

// ==================================================================================================================
// Help Page --------------------------------------------------------------------------------------------------------
// ==================================================================================================================

func (ui *WebUI) OpenHelp() {
	ui.isHelpOpen = true
	ui.helpPage.Style().SetProperty("display", "flex", nil)
	ui.helpPage.Focus(nil)
}

func (ui *WebUI) CloseHelp() {
	ui.isHelpOpen = false
	ui.helpPage.Style().SetProperty("display", "none", nil)

	switch ui.state {
	case StateSplashScreen:
		ui.splashDiv.Focus(nil)
	case StateConfig:
		ui.gameStartDialog.Focus(nil)
	case StateGame:
		ui.canvas.Focus(nil)
	default:
		// No focus.
	}
}
