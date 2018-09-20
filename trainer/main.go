package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	ai_players "github.com/janpfeifer/hiveGo/ai/players"
	"github.com/janpfeifer/hiveGo/ai/tensorflow"
	"github.com/janpfeifer/hiveGo/ascii_ui"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = fmt.Printf

var (
	flag_cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")

	flag_players = [2]*string{
		flag.String("ai0", "", "Configuration string for ai playing as the starting player."),
		flag.String("ai1", "", "Configuration string for ai playing as the second player."),
	}

	flag_maxMoves = flag.Int(
		"max_moves", 100, "Max moves before game is assumed to be a draw.")

	flag_numMatches = flag.Int("num_matches", 1, "Number of matches to play. If larger "+
		"than one, starting position is alternated.")
	flag_print       = flag.Bool("print", false, "Print board at the end of the match.")
	flag_saveMatches = flag.String("save_matches", "", "File name where to save matches.")
	flag_loadMatches = flag.String("load_matches", "",
		"Instead of actually playing matches, load pre-generated ones.")
	flag_wins     = flag.Bool("wins", false, "Counts only matches with wins.")
	flag_winsOnly = flag.Bool("wins_only", false, "Counts only matches with wins (like --wins) and discards draws.")

	flag_lastActions = flag.Int("last_actions", 0, "If set > 0, on the given number of last moves of each match are used for training.")
	flag_train       = flag.Bool("train", false, "Set to true to train with match data.")
	flag_trainLoops  = flag.Int("train_loops", 1, "After acquiring data for all matches, how many times to loop the training over it.")
	flag_rescore     = flag.Int("rescore", 0,
		"If to rescore loaded matches. A value higher than 1 means that it will loop "+
			"over rescoring and retraining.")

	players = [2]*ai_players.SearcherScorePlayer{nil, nil}
)

func init() {
	flag.BoolVar(&tensorflow.CpuOnly, "cpu", false, "Force to use CPU, even if GPU is available")
}

// Results and if the players were swapped.
type Match struct {
	// Wether p0/p1 swapped positions in this match.
	Swapped bool

	// Match actions, alternating players.
	Actions []Action

	// All board states of the game: 1 more than the number of actions.
	Boards []*Board

	// Scores for each board position. Can either be calculated during
	// the match, or re-genarated when re-loading a match.
	Scores []float32
}

func (m *Match) FinalBoard() *Board { return m.Boards[len(m.Boards)-1] }

func (m *Match) Encode(enc *gob.Encoder) {
	boards := m.Boards
	m.Boards = []*Board{boards[0].Copy()}
	if err := enc.Encode(m); err != nil {
		log.Panicf("Failed to encode match: %v", err)
	}
	m.Boards = boards
}

func (m *Match) AppendLabeledExamples(examples []ai.LabeledExample) []ai.LabeledExample {
	from := 0
	if *flag_lastActions > 1 && *flag_lastActions < len(m.Actions) {
		from = len(m.Actions) - *flag_lastActions
	}
	for ii := from; ii < len(m.Actions); ii++ {
		examples = append(examples, ai.MakeLabeledExample(m.Boards[ii], m.Scores[ii]))
	}
	return examples
}

func MatchDecode(dec *gob.Decoder) (match *Match, err error) {
	match = &Match{}
	err = dec.Decode(match)
	if err != nil {
		return
	}
	board := NewBoard()
	board.MaxMoves = match.Boards[0].MaxMoves
	board.BuildDerived()
	match.Boards[0] = board
	for _, action := range match.Actions {
		board = board.Act(action)
		match.Boards = append(match.Boards, board)
	}
	return
}

func runMatch(matchNum int) *Match {
	swapped := (matchNum%2 == 1)
	board := NewBoard()
	board.MaxMoves = *flag_maxMoves
	match := &Match{Swapped: swapped, Boards: []*Board{board}}
	reorderedPlayers := players
	if swapped {
		reorderedPlayers[0], reorderedPlayers[1] = players[1], players[0]
	}

	// Run match.
	for !board.IsFinished() {
		player := board.NextPlayer
		if swapped {
			player = 1 - player
		}
		glog.V(1).Infof("\n\nMatch %d: player %d at turn %d\n\n", matchNum, player, board.MoveNumber)
		var action Action
		score := float32(0)
		if len(board.Derived.Actions) == 0 {
			// Auto-play skip move.
			action = Action{Piece: NO_PIECE}
			board = board.Act(action)
			if len(board.Derived.Actions) == 0 {
				log.Panicf("No moves to either side!?\n\n%v\n", board)
			}
		} else {
			action, board, score = reorderedPlayers[board.NextPlayer].Play(board)
		}
		match.Actions = append(match.Actions, action)
		match.Boards = append(match.Boards, board)
		match.Scores = append(match.Scores, score)
	}

	if glog.V(1) {
		var msg string
		if match.FinalBoard().Draw() {
			msg = "match was a draw!"
		} else {
			player := match.FinalBoard().Winner()
			if swapped {
				player = 1 - player
			}
			msg = fmt.Sprintf("player %d won!", player)
		}
		glog.V(1).Infof("\n\nMatch %d: finished at turn %d, %s\n\n",
			matchNum, match.FinalBoard().MoveNumber, msg)
	}

	return match
}

// runMatches run --repeat number of matches, and write the resulting matches
// to the given channel.
func runMatches(results chan<- *Match) {
	if *flag_winsOnly {
		*flag_wins = true
	}
	// Run at most GOMAXPROCS simultaneously.
	var wg sync.WaitGroup
	semaphore := make(chan bool, runtime.GOMAXPROCS(0))
	done := false
	wins := 0
	for ii := 0; !done; ii++ {
		wg.Add(1)
		semaphore <- true
		go func(matchNum int) {
			defer wg.Done()
			match := runMatch(matchNum)
			if !match.FinalBoard().Draw() {
				wins++
				if *flag_wins {
					done = done || (wins >= *flag_numMatches)
				}
			}
			<-semaphore
			if *flag_winsOnly && match.FinalBoard().Draw() {
				return
			}
			results <- match
		}(ii)
		if !*flag_wins {
			done = done || (ii+1 >= *flag_numMatches)
		}
	}
	wg.Wait()
	close(results)
}

// Open file for writing. If filename already exists rename it by appending an "~" suffix.
func openWriterAndBackup(filename string) io.Writer {
	if _, err := os.Stat(filename); err == nil {
		err = os.Rename(filename, filename+"~")
		if err != nil {
			log.Printf("Failed to rename '%s' to '%s~': %v", filename, filename, err)
		}
	}
	file, err := os.Create(filename)
	if err != nil {
		log.Panicf("Failed to save file to '%s': %v", filename, err)
	}
	return file
}

// Load matches, and automatically build boards.
func loadMatches(results chan<- *Match) {
	file, err := os.Open(*flag_loadMatches)
	if err != nil {
		log.Panicf("Cannot open '%s' for reading: %v", *flag_loadMatches, err)
	}
	dec := gob.NewDecoder(file)

	// Run at most GOMAXPROCS re-scoring simultaneously.
	for ii := 0; true; ii++ {
		match, err := MatchDecode(dec)
		if err == io.EOF {
			break
		}
		if err != nil {
			glog.Errorf("Cannot read any more matches: %v", err)
			break
		}
		if *flag_winsOnly && !match.FinalBoard().Draw() {
			continue
		}
		results <- match
	}
	close(results)
}

func main() {
	flag.Parse()
	if *flag_cpuprofile != "" {
		f, err := os.Create(*flag_cpuprofile)
		if err != nil {
			glog.Fatal("could not create CPU profile: ", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			glog.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *flag_rescore > 0 && !*flag_train {
		log.Fatal("Flag --rescore set, but not --train. Not sure what to do.")
	}
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}
	for ii := 0; ii < 2; ii++ {
		players[ii] = ai_players.NewAIPlayer(*flag_players[ii])
	}

	// Run/load matches.
	results := make(chan *Match)
	if *flag_loadMatches != "" {
		go loadMatches(results)
	} else {
		go runMatches(results)
	}

	if *flag_rescore > 0 {
		loopRescoreAndRetrainMatches(results)
	} else {
		reportMatches(results)
	}
}

func reportMatches(matches chan *Match) {
	// Read results.
	totalWins := [3]int{0, 0, 0}
	totalMoves := 0

	var enc *gob.Encoder
	if *flag_saveMatches != "" {
		file := openWriterAndBackup(*flag_saveMatches)
		enc = gob.NewEncoder(file)
	}

	count := 0
	var labeledExamples []ai.LabeledExample
	ui := ascii_ui.NewUI(true, false)
	for match := range matches {
		count++
		if enc != nil {
			match.Encode(enc)
		}
		if *flag_train {
			labeledExamples = match.AppendLabeledExamples(labeledExamples)
		}

		// Accounting.
		board := match.FinalBoard()
		wins := board.Derived.Wins
		if match.Swapped {
			wins[0], wins[1] = wins[1], wins[0]
		}
		if *flag_print {
			if match.Swapped {
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

	// Train with examples.
	if *flag_train {
		trainFromExamples(labeledExamples)
	}

	// Print totals.
	fmt.Printf("Total matches=%d\n", count)
	for ii, value := range totalWins {
		var p string
		if ii < 2 {
			p = fmt.Sprintf("P%d Wins", ii)
		} else {
			p = "Draws"
		}
		fmt.Printf("%s=%d\t%.1f%%\n", p, value, 100.0*float64(value)/float64(count))
	}
	fmt.Printf("Average number of moves=%.1f\n", float64(totalMoves)/float64(count))
}
