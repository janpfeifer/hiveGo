package main

import (
	"github.com/gopherjs/gopherjs/js"
	"github.com/janpfeifer/hiveGo/ai/players"
	"github.com/janpfeifer/hiveGo/ai/search"
)

var (
	IsAIPlaying, IsAITurn bool
	AIPlayerNum           uint8
	BusyBox               = jq("img#busy")

	aiPlayer *players.SearcherScorerPlayer
)

func StartAI(config string, aiPlayerNum int) {
	IsAIPlaying = true
	AIPlayerNum = uint8(aiPlayerNum)
	aiPlayer = players.NewAIPlayer(config, false)
	ScheduleAIPlay()
	AdjustBusyBoxPosition()
}

func ScheduleAIPlay() {
	if search.IdleChan == nil {
		search.IdleChan = make(chan bool, 1)
	}
	RequestIdleCallback()
	IsAIPlaying = (!Board.IsFinished() && Board.NextPlayer == AIPlayerNum)
	if IsAIPlaying {
		BusyBox.SetCss("display", "block")
		go AIPlay()
	} else {
		BusyBox.SetCss("display", "none")
	}
}

func AdjustBusyBoxPosition() {
	scale := IMAGE_BASE_SIZE * 1025 * ui.PixelRatio
	height, width := int(scale), int(1.5*scale)
	SetAttrs(Obj(BusyBox), Attrs{
		"width":  width,
		"height": height,
	})

	top := ui.OffBoardHeight()/2 - height/2
	left := width / 2
	if AIPlayerNum == 1 {
		top = ui.Height - ui.OffBoardHeight()/2 - height/2
	}
	BusyBox.SetCss("top", top)
	BusyBox.SetCss("left", left)
}

func AIPlay() {
	action, _, _ := aiPlayer.Play(Board)
	ExecuteAction(action)
}

var waitingIdleProcessing = false

func IdleCallback() {
	if search.IdleChan == nil {
		return
	}

	// Send a signal to process a chunk, but doesn't block. So if there
	// is already a signal in IdleChan another one is not sent.
	select {
	case search.IdleChan <- true:
		// Process a chunk.
	default:
		// Nothing to do.
	}

	if IsAIPlaying {
		// While AI is still thinking, reschedule the callback.
		RequestIdleCallback()
	}
}

func RequestIdleCallback() {
	window.Call("requestIdleCallback", js.MakeFunc(
		func(this *js.Object, arguments []*js.Object) interface{} {
			IdleCallback()
			return nil
		}))
}
