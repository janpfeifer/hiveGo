package main

import (
	"fmt"
	"log"

	"github.com/gotk3/gotk3/cairo"
	"github.com/gotk3/gotk3/gdk"
	"github.com/gotk3/gotk3/glib"
	"github.com/gotk3/gotk3/gtk"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = fmt.Printf

var (
	mainWindow      *gtk.Window
	mainDrawing     *gtk.DrawingArea
	offBoardDrawing [2]*gtk.DrawingArea
	cairoCtx        *cairo.Context
)

var (
	// Dragging postion: values valid after button down.
	isDragging    = false
	dragStartTime uint32
	dragX, dragY  float64
)

// Amount of time below which a button press is considered a click.
// After that it is consider a click-and-hold (or drag).
const CLICK_MAX_TIME_MS = 500

// Creates the main window.
func createMainWindow() {
	// Create a new toplevel window, set its title, and connect it to the
	// "destroy" signal to exit the GTK main loop when it is destroyed.
	win, err := gtk.WindowNew(gtk.WINDOW_TOPLEVEL)
	mainWindow = win
	if err != nil {
		log.Fatal("Unable to create window:", err)
	}
	win.SetTitle("gnome-hive")
	win.Connect("destroy", func() {
		gtk.MainQuit()
	})

	createHeaderWithMenu(win)
	createAccelGroup(win)
	loadImageResources()

	// Off board drawing area for players.
	for ii := 0; ii < 2; ii++ {
		offBoardDrawing[ii], err = gtk.DrawingAreaNew()
		if err != nil {
			log.Fatal("Unable to create DrawingArea:", err)
		}
		offBoardDrawing[ii].SetSizeRequest(800, 100)
		offBoardDrawing[ii].AddEvents(int(gdk.BUTTON_PRESS_MASK))

		player := uint8(ii)
		offBoardDrawing[ii].Connect("draw", func(da *gtk.DrawingArea, cr *cairo.Context) {
			drawOffBoardArea(da, cr, player)
		})
		offBoardDrawing[ii].Connect("button-press-event", func(da *gtk.DrawingArea, ev *gdk.Event) bool {
			evB := &gdk.EventButton{ev}
			if evB.Button() != 1 {
				// We are only interested in the primary button.
				return false
			}
			if !started || finished {
				return false
			}
			if hasSelectedPiece {
				hasSelectedPiece = false
				mainWindow.QueueDraw() // Needs redrawing.
			}
			if player != board.NextPlayer {
				// Not this players turn, so nothing to select. Just in case, de-seleect.
				if selectedOffBoardPiece != NO_PIECE {
					selectedOffBoardPiece = NO_PIECE
					mainWindow.QueueDraw() // Needs redrawing.
				}
				return true
			}

			// Check where was the click:
			x, y := evB.X(), evB.Y()
			piece := offBoardPositionToPiece(da, x, y)
			if piece != selectedOffBoardPiece {
				selectedOffBoardPiece = piece
				mainWindow.QueueDraw()
			}
			return true
		})
	}

	// Main drawing area for board.
	mainDrawing, err = gtk.DrawingAreaNew()
	if err != nil {
		log.Fatal("Unable to create DrawingArea:", err)
	}
	mainDrawing.Connect("draw", drawMainBoard)
	mainDrawing.AddEvents(int(
		gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK | gdk.POINTER_MOTION_MASK |
			gdk.SCROLL_MASK))
	mainDrawing.Connect("configure-event", func(da *gtk.DrawingArea, ev *gdk.Event) bool {
		// Resize means redrawing full window.
		mainDrawing.QueueDraw()
		return false
	})
	mainDrawing.Connect("button-press-event", func(da *gtk.DrawingArea, ev *gdk.Event) bool {
		evB := &gdk.EventButton{ev}
		if evB.Button() != 1 {
			return false
		}
		isDragging = true
		dragStartTime = evB.Time()
		dragX, dragY = evB.X(), evB.Y()
		return true
	})
	mainDrawing.Connect("button-release-event", func(da *gtk.DrawingArea, ev *gdk.Event) bool {
		evB := &gdk.EventButton{ev}
		if evB.Button() != 1 {
			return false
		}
		isDragging = false
		if evB.Time()-dragStartTime <= CLICK_MAX_TIME_MS {
			if !started {
				newGame()
				return true
			}
			if !finished {
				mainBoardClick(mainDrawing, evB.X(), evB.Y())
			}
			return true
		}
		return true
	})
	mainDrawing.Connect("motion-notify-event", func(da *gtk.DrawingArea, ev *gdk.Event) bool {
		evM := &gdk.EventMotion{ev}
		x, y := evM.MotionVal()
		if !isDragging {
			return false
		}
		// TODO: check that CLICK_MAX_TIME_MS has elapsed before starting drag.
		deltaX, deltaY := x-dragX, y-dragY
		dragX, dragY = x, y
		shiftX += deltaX
		shiftY += deltaY
		mainDrawing.QueueDraw()
		return true

	})
	mainDrawing.Connect("scroll-event", func(da *gtk.DrawingArea, ev *gdk.Event) bool {
		evS := &gdk.EventScroll{ev}
		dir := evS.Direction()
		if dir == gdk.SCROLL_UP {
			zoomFactor *= 1.2
		} else if dir == gdk.SCROLL_DOWN {
			zoomFactor /= 1.2
		} else {
			return false
		}
		mainDrawing.QueueDraw()
		return true
	})

	// Assemble main window area.
	box, err := gtk.BoxNew(gtk.ORIENTATION_VERTICAL, 3)
	if err != nil {
		log.Fatal("Unable to create box:", err)
	}
	box.PackStart(offBoardDrawing[0], false, true, 0)
	box.PackStart(mainDrawing, true, true, 0)
	box.PackStart(offBoardDrawing[1], false, true, 0)
	win.Add(box)

	// Set the default window size.
	win.SetDefaultSize(800, 600)
}

func createHeaderWithMenu(win *gtk.Window) {
	// Create a header bar.
	header, err := gtk.HeaderBarNew()
	if err != nil {
		log.Fatal("Could not create header bar:", err)
	}
	header.SetShowCloseButton(true)
	header.SetTitle("gnome-hive")
	header.SetSubtitle("Hive implementation in Go")

	// Create a menu button.
	mbtn, err := gtk.MenuButtonNew()
	if err != nil {
		log.Fatal("Could not create menu button:", err)
	}
	menu := glib.MenuNew()
	if menu == nil {
		log.Fatal("Could not create menu (nil)")
	}
	menu.Append("New Game - ctrl+N", "win.new_game")
	menu.Append("Quit - ctrl+Q", "win.quit")
	mbtn.SetMenuModel(&menu.MenuModel)
	header.PackStart(mbtn)
	win.SetTitlebar(header)

	// Register actions.
	aQuit := glib.SimpleActionNew("quit", nil)
	aQuit.Connect("activate", func() { win.Close() })

	aNewGame := glib.SimpleActionNew("new_game", nil)
	aNewGame.Connect("activate", func() {
		newGame()
	})

	actG := glib.SimpleActionGroupNew()
	actG.AddAction(aQuit)
	actG.AddAction(aNewGame)
	win.InsertActionGroup("win", actG)
}

func createAccelGroup(win *gtk.Window) {
	accelG, err := gtk.AccelGroupNew()
	if err != nil {
		log.Fatal("Could not create menu button:", err)
	}
	key, mods := gtk.AcceleratorParse("<Control>Q")
	accelG.Connect(key, mods, gtk.ACCEL_VISIBLE, func() { mainWindow.Close() })
	key, mods = gtk.AcceleratorParse("<Control>N")
	accelG.Connect(key, mods, gtk.ACCEL_VISIBLE, func() {
		newGame()
	})
	win.AddAccelGroup(accelG)
}

func mainBoardClick(da *gtk.DrawingArea, x, y float64) {
	dp := newDrawingParams(mainDrawing)
	pos := dp.XYToPos(x, y)
	if selectedOffBoardPiece != NO_PIECE {
		if _, ok := placementPositions()[pos]; ok {
			// Placement action selected, execute it.
			for _, action := range board.Derived.Actions {
				if !action.Move && action.Piece == selectedOffBoardPiece && action.TargetPos == pos {
					executeAction(action)
					return
				}
			}
			selectedOffBoardPiece = NO_PIECE
			return
		} else {
			selectedOffBoardPiece = NO_PIECE
			mainWindow.QueueDraw()
		}
	}

	if hasSelectedPiece {
		for _, action := range board.Derived.Actions {
			if action.Move && action.SourcePos == selectedPiecePos {
				if action.TargetPos == pos {
					executeAction(action)
					return
				}
			}
		}

		// If clicked somewhere else, deselect current piece and continue (maybe it will
		// select another piece).
		hasSelectedPiece = false
		mainWindow.QueueDraw()
	}

	// Check if a piece has been selected.
	if _, ok := moveSourcePositions()[pos]; ok {
		hasSelectedPiece = true
		selectedPiecePos = pos
		mainWindow.QueueDraw()
	}
}

func placementPositions() (posMap map[Pos]bool) {
	posMap = make(map[Pos]bool)
	for _, action := range board.Derived.Actions {
		if !action.Move {
			posMap[action.TargetPos] = true
		}
	}
	return
}

func moveSourcePositions() (posMap map[Pos]bool) {
	posMap = make(map[Pos]bool)
	for _, action := range board.Derived.Actions {
		if action.Move {
			posMap[action.SourcePos] = true
		}
	}
	return
}
