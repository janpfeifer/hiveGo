package main

import (
	"flag"
	"fmt"
	"log"

	// TensorFlow is included so it shows up as an option for scorers.
	_ "github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ai/players"
	//_ "github.com/janpfeifer/hiveGo/ai/tensorflow"
	_ "github.com/janpfeifer/hiveGo/ai/search/ab"
	_ "github.com/janpfeifer/hiveGo/ai/search/mcts"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var (
	_ = fmt.Printf

	flag_players = [2]*string{
		flag.String("p0", "ai", "First player: hotseat, ai"),
		flag.String("p1", "hotseat", "First player: hotseat, ai"),
	}
	flag_aiConfig = flag.String("ai", "", "Configuration of AI.")
	flag_aiUI     = flag.Bool("ai_ui", false, "Shows UI even for ai vs ai game.")
	flag_maxMoves = flag.Int(
		"max_moves", 200, "Max moves before game is assumed to be a draw.")

	aiPlayers = [2]players.Player{nil, nil}
)

func main() {
	flag.Parse()
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}

	// Create players:
	for ii := 0; ii < 2; ii++ {
		switch {
		case *flag_players[ii] == "hotseat":
			continue
		case *flag_players[ii] == "ai":
			aiPlayers[ii] = players.NewAIPlayer(*flag_aiConfig, true)
		default:
			log.Fatalf("Unknown player type --p%d=%s", ii, *flag_players[ii])
		}
	}

	board := NewBoard()
	board.MaxMoves = *flag_maxMoves

	ui := ascii_ui.NewUI(true, false)

	for !board.IsFinished() {
		if newBoard, skip := ui.CheckNoAvailableAction(board); skip {
			board = newBoard
			continue
		}
		aiPlayer := aiPlayers[board.NextPlayer]
		if aiPlayer == nil {
			newBoard, err := ui.RunNextMove(board)
			if err != nil {
				log.Fatalf("Failed to run match: %v", err)
			}
			board = newBoard
		} else {
			// AI plays.
			if *flag_aiUI {
				ui.Print(board)
			} else {
				fmt.Println()
				ui.PrintPlayer(board)
			}
			action, newBoard, _, _ := aiPlayer.Play(board, fmt.Sprintf("AI_P%d", board.NextPlayer))
			fmt.Printf("  Action for player %d: %s\n", board.NextPlayer, action)
			board = newBoard
			fmt.Println()
		}
	}
	ui.PrintWinner(board)

}
