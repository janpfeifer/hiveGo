package main

import (
	"github.com/gopherjs/gopherjs/js"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/state"

	_ "github.com/janpfeifer/hiveGo/internal/players/default"
)

var (
	AIPlayerNum state.PlayerNum
	BusyBox     = jq("img#busy")
)

// CooperativeConcurrency is an optional interface that Searchers can implement to play nicely
// with the browser (not freeze it).
type CooperativeConcurrency interface {
	// SetIdleChan sets a channel that should be listened to before every chunk of work
	// is done. It works as a time-sharing mechanism between the browser and the Go code.
	SetIdleChan(idleChan <-chan bool)
}

// StartAI creates the AI player.
func (g *Game) StartAI(config string, aiPlayerNum state.PlayerNum) error {
	g.IsAIPlaying = true
	AIPlayerNum = aiPlayerNum
	var err error
	g.aiPlayer, err = players.New(config)
	if err != nil {
		return err
	}

	if cooperative, ok := g.aiPlayer.Searcher.(CooperativeConcurrency); ok {
		g.isCooperative = true
		g.idleChan = make(chan bool, 1)
		cooperative.SetIdleChan(g.idleChan)
		g.RequestIdleCallback()
	}

	g.ScheduleAIPlay()
	AdjustBusyBoxPosition()
	return nil
}

func (g *Game) ScheduleAIPlay() {
	if g.isCooperative {
		g.RequestIdleCallback()
	}
	g.IsAIPlaying = (!g.board.IsFinished() && g.board.NextPlayer == AIPlayerNum)
	if g.IsAIPlaying {
		BusyBox.SetCss("display", "block")
		go g.AIPlay()
	} else {
		BusyBox.SetCss("display", "none")
	}
}

func AdjustBusyBoxPosition() {
	scale := ImageBaseSize * 1025 * ui.PixelRatio
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

func (g *Game) AIPlay() {
	action, _, _, _ := g.aiPlayer.Play(g.board)
	g.ExecuteAction(action)
}

var waitingIdleProcessing = false

func (g *Game) IdleCallback() {
	if !g.isCooperative {
		return
	}

	// Send a signal to process a chunk but doesn't block.
	select {
	case g.idleChan <- true:
		// Process a chunk.
	default:
		// Nothing to do: nobody is listening on the other side.
	}

	if g.IsAIPlaying {
		// While AI is still thinking, reschedule the callback.
		g.RequestIdleCallback()
	}
}

func (g *Game) RequestIdleCallback() {
	window.Call("requestIdleCallback", js.MakeFunc(
		func(this *js.Object, arguments []*js.Object) interface{} {
			g.IdleCallback()
			return nil
		}))
}
