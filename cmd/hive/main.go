package main

import (
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
)

var (
	_ = fmt.Printf

	flagHotseat   = flag.Bool("hotseat", false, "Hotseat match: human vs human")
	flagWatch     = flag.Bool("watch", false, "Watch mode: AI vs AI playing")
	flagFirst     = flag.String("first", "", "Who plays first: human or ai. Default is random.")
	flagAIConfig  = flag.String("config", "linear:ab", "AI configuration against which to play")
	flagAIConfig2 = flag.String("config2", "linear:ab", "Second AI configuration, if playing AI vs AI with --watch")
	flagAIWithUI  = flag.Bool("ai_ui", false, "Shows UI even for ai vs ai game.")
	flagMaxMoves  = flag.Int(
		"max_moves", DefaultMaxMoves, "Max moves before game is considered a draw.")

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
			if *flagAIWithUI {
				ui.Print(board)
			} else {
				fmt.Println()
				ui.PrintPlayer(board)
			}
			action, newBoard, score, _ := aiPlayer.Play(board)
			fmt.Printf("  Action for %s player: %s (score=%.3f)\n", board.NextPlayer, action, score)
			board = newBoard
			fmt.Println()
		}
	}
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
