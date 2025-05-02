package main

import (
	"fmt"
	"github.com/gowebapi/webapi"
	"github.com/gowebapi/webapi/html/htmlevent"
)

// HTML page core elements:
var (
	Document = webapi.GetDocument()
	Window   = webapi.GetWindow()
)

func main() {
	ui := NewWebUI()
	fmt.Printf("UI: %+v\n", ui)
	ui.CreateSplashScreen()
	Window.SetOnResize(func(event *htmlevent.UIEvent, currentTarget *webapi.Window) {
		ui.OnCanvasResize()
	})
	//ui.SetBusy(true)

	// Wait forever: the Wasm program will never exit, while the page is opened.
	select {}
}
