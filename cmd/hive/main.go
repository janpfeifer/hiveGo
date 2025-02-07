package main

import (
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/players"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"k8s.io/klog/v2"
	"log"
)

var (
	_ = fmt.Printf

	//flagPlayers = [2]*string{
	//	flag.String("p0", "hotseat", "First player: hotseat, ai"),
	//	flag.String("p1", "hotseat", "First player: hotseat, ai"),
	//}
	//flag_aiConfig = flag.String("ai", "", "Configuration of AI.")
	//flagAIWithUI     = flag.Bool("ai_ui", false, "Shows UI even for ai vs ai game.")
	flagMaxMoves = flag.Int(
		"max_moves", 200, "Max moves before game is considered a draw.")

	aiPlayers = [2]players.Player{nil, nil}
)

func main() {
	flag.Parse()
	if *flagMaxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flagMaxMoves)
	}

	// Create players:
	//for ii := 0; ii < 2; ii++ {
	//	switch {
	//	case *flagPlayers[ii] == "hotseat":
	//		continue
	//	case *flagPlayers[ii] == "ai":
	//		klog.Exitf("AI player not implemented yet.")
	//		klog.V(1).Infof("Creating AI player with config %s", *flag_aiConfig)
	//		//aiPlayers[ii] = players.New(*flag_aiConfig, true)
	//	default:
	//		log.Fatalf("Unknown player type --p%d=%s", ii, *flagPlayers[ii])
	//	}
	//}

	board := NewBoard()
	board.MaxMoves = *flagMaxMoves

	ui := cli.New(true, false)

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
			//} else {
			//	// AI plays.
			//	if *flagAIWithUI {
			//		ui.Print(board)
			//	} else {
			//		fmt.Println()
			//		ui.PrintPlayer(board)
			//	}
			//	action, newBoard, _, _ := aiPlayer.Play(board, fmt.Sprintf("AI_P%d", board.NextPlayer))
			//	fmt.Printf("  Action for player %d: %s\n", board.NextPlayer, action)
			//	board = newBoard
			//	fmt.Println()
		}
	}
	ui.PrintWinner(board)
}
