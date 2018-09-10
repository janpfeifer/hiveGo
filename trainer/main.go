package main

import (
	"flag"
	"fmt"
	"log"

	ai_players "github.com/janpfeifer/hiveGo/ai/players"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = fmt.Printf

var (
	flag_players = [2]*string{
		flag.String("ai0", "", "Configuration string for ai playing as the starting player."),
		flag.String("ai1", "", "Configuration string for ai playing as the second player."),
	}

	flag_maxMoves = flag.Int(
		"max_moves", 200, "Max moves before game is assumed to be a draw.")

	flag_repeats = flag.Int("repeats", 1, "Number of times to repeat the game. If larger "+
		"than one, starting position is alternated.")
	flag_print = flag.Bool("print", false, "Print board at the end of the match.")

	players = [2]ai_players.Player{nil, nil}
)

func main() {
	flag.Parse()
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}
	ui := ascii_ui.NewUI(true, false)
	for ii := 0; ii < 2; ii++ {
		players[ii] = ai_players.NewAIPlayer(*flag_players[ii])
	}

	// Results and if the players were swapped.
	results := make(chan *Board, 1)
	swapped := make(chan bool, 1)
	for ii := 0; ii < *flag_repeats; ii++ {
		go func(match int) {
			board := NewBoard()
			board.MaxMoves = *flag_maxMoves
			reorderedPlayers := players
			if ii%2 == 1 {
				reorderedPlayers[0], reorderedPlayers[1] = players[1], players[0]
			}

			// Run match.
			for !board.IsFinished() {
				if len(board.Derived.Actions) == 0 {
					// Auto-play skip move.
					board = board.Act(Action{Piece: NO_PIECE})
					if len(board.Derived.Actions) == 0 {
						ui.PrintBoard(board)
						log.Panicf("No moves to either side!?")
					}
				}
				action := reorderedPlayers[board.NextPlayer].Play(board)
				board = board.Act(action)
			}
			results <- board
			swapped <- ii%2 == 1
		}(ii)
	}

	// Read results.
	totalWins := [3]int{0, 0, 0}
	totalMoves := 0
	for ii := 0; ii < *flag_repeats; ii++ {
		board := <-results
		wins := board.Derived.Wins
		if <-swapped {
			wins[0], wins[1] = wins[1], wins[0]
		}
		if *flag_print {
			ui.PrintBoard(board)
			ui.PrintWinner(board)
			fmt.Println()
			fmt.Println()
		}
		if wins[0] == wins[1] {
			totalWins[2]++
		} else if wins[0] {
			totalWins[0]++
		} else {
			totalWins[1]++
		}
		totalMoves += board.MoveNumber
	}

	// Print totals.
	fmt.Printf("Total matches=%d\n", *flag_repeats)
	for ii, value := range totalWins {
		var p string
		if ii < 2 {
			p = fmt.Sprintf("P%d Wins", ii)
		} else {
			p = "Draws"
		}
		fmt.Printf("%s=%d\t%.1f%%\n", p, value, 100.0*float64(value)/float64(*flag_repeats))
	}
	fmt.Printf("Average number of moves=%.1f\n", float64(totalMoves)/float64(*flag_repeats))
}
