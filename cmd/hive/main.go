package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/janpfeifer/hiveGo/internal/players"
	_ "github.com/janpfeifer/hiveGo/internal/players/default"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"math/rand/v2"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

var (
	_ = fmt.Printf

	flagHotseat   = flag.Bool("hotseat", false, "Hotseat match: human vs human")
	flagWatch     = flag.Bool("watch", false, "Watch mode: AI vs AI playing")
	flagFirst     = flag.String("first", "", "Who plays first: human or ai. Default is random.")
	flagAIConfig  = flag.String("config", "linear,ab", "AI configuration against which to play")
	flagAIConfig2 = flag.String("config2", "linear,ab", "Second AI configuration, if playing AI vs AI with --watch")
	flagMaxMoves  = flag.Int(
		"max_moves", DefaultMaxMoves, "Max moves before game is considered a draw.")
	flagQuiet = flag.Bool("quiet", false, "Quiet mode for when watching AI play, only the actions and the last board position is printed.")

	// aiPlayers: if nil, it's a human playing.
	aiPlayers = [2]players.Player{nil, nil}
	matchId   = uint64(0)
	matchName = "The Match"

	globalCtx = context.Background()
)

func main() {
	flag.Parse()
	if *flagMaxMoves <= 0 {
		klog.Fatalf("Invalid --max_moves=%d", *flagMaxMoves)
	}

	// Capture Control+C
	var cancel func()
	globalCtx, cancel = context.WithCancel(context.Background())
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		cancel()
		fmt.Print("\033[?25h\033[39;49;0m\n") // Restore cursor and colors.
		klog.Fatalf("Got interrupt, shutting down...")
	}()

	// Create players.
	createPlayers()

	// Create board and UI.
	board := NewBoard()
	board.MaxMoves = *flagMaxMoves
	ui := cli.New(true, false)

	// Loop over match.
	for !board.IsFinished() {
		if newBoard, skip := ui.CheckNoAvailableAction(board); skip {
			board = newBoard
			continue
		}
		aiPlayer := aiPlayers[board.NextPlayer]
		if aiPlayer == nil {
			newBoard, err := ui.RunNextMove(board)
			if err != nil {
				klog.Exitf("Failed to run match: %+v", err)
			}
			board = newBoard
		} else {
			// AI plays.
			if *flagWatch && !*flagQuiet {
				ui.Print(board, false)
				fmt.Print("\tAction: ")
			} else {
				ui.PrintSpacedPlayer(board)
				fmt.Print(": ")
			}

			s := NewSpinning()
			action, newBoard, score, _ := aiPlayer.Play(board)
			s.Done()
			fmt.Printf(" %s (score=%.3f)\n", action, score)
			board = newBoard
			fmt.Println()
		}
	}

	ui.Print(board, false)
	ui.PrintWinner(board)
}

// createPlayers in aiPlayers.
func createPlayers() {
	if *flagHotseat && *flagWatch {
		klog.Fatalf("--hotseat and --watch cannot be used together")
	}
	if *flagHotseat {
		// Both players are human, nothing to do.
		return
	}

	// Create AI player:
	var aiPlayerNum PlayerNum
	if *flagWatch {
		aiPlayerNum = 0
	} else {
		if strings.ToLower(*flagFirst) == "human" {
			aiPlayerNum = 1
		} else if strings.ToLower(*flagFirst) == "ai" {
			aiPlayerNum = 0
		} else if *flagFirst == "" {
			// Random:
			aiPlayerNum = PlayerNum(rand.IntN(2))
		} else {
			exceptions.Panicf("invalid --first=%q, only valid values are \"human\" or \"ai\"", *flagFirst)
		}
	}
	aiPlayers[aiPlayerNum] = must.M1(players.New(*flagAIConfig))
	if !*flagWatch {
		return
	}

	// Create second AI
	otherPlayerNum := 1 - aiPlayerNum
	aiPlayers[otherPlayerNum] = must.M1(players.New(*flagAIConfig2))
	return
}

type Spinning struct {
	wg     sync.WaitGroup
	cancel func()
}

var (
	//spinningSeq = []rune("`|/-\`")
	//spinningSeq = []rune("🌑🌒🌓🌔🌕🌖🌗🌘") // `|/-\`
	spinningSeq = []rune("🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛") // `|/-\`
	spinningIdx int
	spinningLen = len(spinningSeq)
)

func NewSpinning() *Spinning {
	s := &Spinning{}
	var ctx context.Context
	ctx, s.cancel = context.WithCancel(globalCtx)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		ticker := time.NewTicker(500 * time.Millisecond)
		fmt.Print("\033[?25l")       // Hide cursor.
		defer fmt.Print("\033[?25h") // Restore cursor.

		fmt.Print("  ")
		for {
			symbol := spinningSeq[spinningIdx]
			fmt.Printf("\b\b%c", symbol)
			spinningIdx = (spinningIdx + 1) % spinningLen
			select {
			case <-ctx.Done():
				fmt.Print("\b\b")
				return
			case <-ticker.C:
				// continue
			}
		}
	}()
	return s
}

func (s *Spinning) Done() {
	s.cancel()
	s.wg.Wait()
}
