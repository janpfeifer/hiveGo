package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"

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
		"max_moves", 100, "Max moves before game is assumed to be a draw.")

	flag_repeats = flag.Int("repeats", 1, "Number of times to repeat the game. If larger "+
		"than one, starting position is alternated.")
	flag_print = flag.Bool("print", false, "Print board at the end of the match.")

	players = [2]*ai_players.SearcherScorePlayer{nil, nil}
)

// Results and if the players were swapped.
type Match struct {
	// Wether p0/p1 swapped positions in this match.
	swapped bool

	// Match actions, alternating players.
	actions []Action

	// All board states of the game: 1 more than the number of actions.
	boards []*Board

	// Scores for each board position. Can either be calculated during
	// the match, or re-genarated when re-loading a match.
	scores []float64
}

func (m *Match) FinalBoard() *Board { return m.boards[len(m.boards)-1] }

func runMatch(matchNum int) *Match {
	swapped := (matchNum%2 == 1)
	board := NewBoard()
	board.MaxMoves = *flag_maxMoves
	match := &Match{swapped: swapped, boards: []*Board{board}}
	reorderedPlayers := players
	if swapped {
		reorderedPlayers[0], reorderedPlayers[1] = players[1], players[0]
	}

	// Run match.
	for !board.IsFinished() {
		log.Printf("\n\nMatch %d: turn %d\n\n", matchNum, board.MoveNumber)
		var action Action
		var nextBoard *Board
		score := 0.0
		if len(board.Derived.Actions) == 0 {
			// Auto-play skip move.
			action = Action{Piece: NO_PIECE}
			nextBoard = board.Act(action)
			if len(board.Derived.Actions) == 0 {
				log.Panicf("No moves to either side!?\n\n%v\n", board)
			}
		} else {
			action, nextBoard, score = reorderedPlayers[board.NextPlayer].Play(board)
		}
		match.actions = append(match.actions, action)
		match.boards = append(match.boards, nextBoard)
		match.scores = append(match.scores, score)
	}
	log.Printf("\n\nMatch %d: finished at turn %d\n\n",
		matchNum, match.FinalBoard().MoveNumber)

	return match
}

// runMatches run --repeat number of matches, and write the resulting matches
// to the given channel.
func runMatches(results chan<- *Match) {
	// Run at most GOMAXPROCS simultaneously.
	semaphore := make(chan bool, runtime.GOMAXPROCS(0))
	for ii := 0; ii < *flag_repeats; ii++ {
		semaphore <- true
		go func(matchNum int) {
			match := runMatch(matchNum)
			<-semaphore
			results <- match
		}(ii)
	}
}

func main() {
	flag.Parse()
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}
	ui := ascii_ui.NewUI(true, false)
	for ii := 0; ii < 2; ii++ {
		players[ii] = ai_players.NewAIPlayer(*flag_players[ii])
	}

	// Run/load matches.
	results := make(chan *Match)
	runMatches(results)

	// Read results.
	totalWins := [3]int{0, 0, 0}
	totalMoves := 0
	for ii := 0; ii < *flag_repeats; ii++ {
		match := <-results
		board := match.FinalBoard()
		wins := board.Derived.Wins
		if match.swapped {
			wins[0], wins[1] = wins[1], wins[0]
		}
		if *flag_print {
			if match.swapped {
				fmt.Printf("*** Players swapped positions at this match! ***\n")
			}
			ui.PrintBoard(board)
			ui.PrintWinner(board)
			fmt.Println()
			fmt.Println()
		}
		if board.Draw() {
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
