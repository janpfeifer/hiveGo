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
	"strings"
	"sync"
	"time"
)

var (
	_ = fmt.Printf

	flagHotseat   = flag.Bool("hotseat", false, "Hotseat match: human vs human")
	flagWatch     = flag.Bool("watch", false, "Watch mode: AI vs AI playing")
	flagFirst     = flag.String("first", "", "Who plays first: human or ai. Default is random.")
	flagAIConfig  = flag.String("config", "linear:ab", "AI configuration against which to play")
	flagAIConfig2 = flag.String("config2", "linear:ab", "Second AI configuration, if playing AI vs AI with --watch")
	flagMaxMoves  = flag.Int(
		"max_moves", DefaultMaxMoves, "Max moves before game is considered a draw.")
	flagQuiet = flag.Bool("quiet", false, "Quiet mode for when watching AI play, only the actions and the last board position is printed.")

	// aiPlayers: if nil, it's a human playing.
	aiPlayers = [2]players.Player{nil, nil}
	matchId   = uint64(0)
	matchName = "The Match"
)

func main() {
	flag.Parse()
	if *flagMaxMoves <= 0 {
		klog.Fatalf("Invalid --max_moves=%d", *flagMaxMoves)
	}

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
	aiPlayers[aiPlayerNum] = must.M1(players.New(matchId, matchName, aiPlayerNum, *flagAIConfig))
	if !*flagWatch {
		return
	}

	// Create second AI
	otherPlayerNum := 1 - aiPlayerNum
	aiPlayers[otherPlayerNum] = must.M1(players.New(matchId, matchName, otherPlayerNum, *flagAIConfig2))
	return
}

type Spinning struct {
	wg     sync.WaitGroup
	cancel func()
}

func NewSpinning() *Spinning {
	const spinningSeq = `|/-\`
	s := &Spinning{}
	var ctx context.Context
	ctx, s.cancel = context.WithCancel(context.Background())
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		ticker := time.NewTicker(500 * time.Millisecond)
		idx := 0
		for {
			fmt.Printf("%c\033[1D", spinningSeq[idx])
			idx = (idx + 1) % len(spinningSeq)
			select {
			case <-ctx.Done():
				fmt.Print("\033[1D")
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
