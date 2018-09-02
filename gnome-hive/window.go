package main

import (
	"fmt"
	"log"

	"github.com/gotk3/gotk3/cairo"
	"github.com/gotk3/gotk3/glib"
	"github.com/gotk3/gotk3/gtk"
	// . "github.com/janpfeifer/hiveGo/state"
)

var _ = fmt.Printf

var (
	mainWindow      *gtk.Window
	mainDrawing     *gtk.DrawingArea
	offBoardDrawing [2]*gtk.DrawingArea
	cairoCtx        *cairo.Context
)

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
		iiCopy := ii
		offBoardDrawing[ii].Connect("draw", func(da *gtk.DrawingArea, cr *cairo.Context) {
			drawOffBoardArea(da, cr, uint8(iiCopy))
		})
	}

	// Main drawing area for board.
	mainDrawing, err = gtk.DrawingAreaNew()
	if err != nil {
		log.Fatal("Unable to create DrawingArea:", err)
	}
	mainDrawing.Connect("draw", drawMainBoard)

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

var (
	x, y float64
)

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
		log.Printf("win.new_game activated!")
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
		log.Printf("win.new_game activated!")
	})
	win.AddAccelGroup(accelG)
}
