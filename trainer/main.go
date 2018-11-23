package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/ai"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sync"

	"github.com/golang/glog"
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

	flag_numMatches = flag.Int("num_matches", 0, "Number of matches to play. If larger "+
		"than one, starting position is alternated. Value of 0 means 1 match to play, or load all file.")
	flag_print       = flag.Bool("print", false, "Print board at the end of the match.")
	flag_printSteps  = flag.Bool("print_steps", false, "Print board at each step.")
	flag_saveMatches = flag.String("save_matches", "", "File name where to save matches.")
	flag_loadMatches = flag.String("load_matches", "",
		"Instead of actually playing matches, load pre-generated ones.")
	flag_loadOnlyMatch = flag.Int("match_idx", -1, "If set it will only load this one "+
		"specific match, useful for debugging.")
	flag_wins     = flag.Bool("wins", false, "Counts only matches with wins.")
	flag_winsOnly = flag.Bool("wins_only", false, "Counts only matches with wins (like --wins) and discards draws.")

	flag_lastActions = flag.Int("last_actions", 0, "If set > 0, on the given number of last moves of each match are used for training.")
	flag_train       = flag.Bool("train", false, "Set to true to train with match data.")
	flag_trainLoops  = flag.Int("train_loops", 1, "After acquiring data for all matches, how many times to loop the training over it.")
	flag_rescore     = flag.Int("rescore", 0,
		"If to rescore loaded matches. A value higher than 1 means that it will loop "+
			"over rescoring and retraining.")
	flag_learningRate = flag.Float64("learning_rate", 1e-5, "Learning rate when learning")

	flag_parallelism = flag.Int("parallelism", 0, "If > 0 ignore GOMAXPROCS and play "+
		"these many matches simultaneously.")
	flag_maxAutoBatch = flag.Int("max_auto_batch", 0, "If > 0 ignore at most do given value of "+
		"auto-batch for tensorflow evaluations.")

	players = [2]*ai_players.SearcherScorerPlayer{nil, nil}
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

	// Probability distributions over actions, to learn from.
	// For AlphaBetaPruning these will be one-hot-encoding of the action taken. But
	// for MCTS (alpha-zero algorithm) these may be different.
	ActionsLabels [][]float32

	// All board states of the game: 1 more than the number of actions.
	Boards []*Board

	// Scores for each board position. Can either be calculated during
	// the match, or re-genarated when re-loading a match.
	Scores []float32

	// Index of the match in the file it was loaded from.
	MatchFileIdx int
}

func (m *Match) FinalBoard() *Board { return m.Boards[len(m.Boards)-1] }

func (m *Match) Encode(enc *gob.Encoder) {
	if err := SaveMatch(enc, m.Boards[0].MaxMoves, m.Actions, m.Scores); err != nil {
		log.Panicf("Failed to encode match: %v", err)
	}
}

// AppendLabeledExamples will add examples for learning _for Player 0 only_.
func (m *Match) AppendLabeledExamples(boardExamples []*Board, boardLabels []float32, actionsLabels [][]float32) (
	[]*Board, []float32, [][]float32) {
	from := 0
	if *flag_lastActions > 0 && *flag_lastActions < len(m.Actions) {
		from = len(m.Actions) - *flag_lastActions
	}
	glog.V(2).Infof("Making LabeledExample, version=%d", players[0].Scorer.Version())
	for ii := from; ii < len(m.Actions); ii++ {
		boardExamples = append(boardExamples, m.Boards[ii])
		boardLabels = append(boardLabels, m.Scores[ii])
		actionsLabels = append(actionsLabels, m.ActionsLabels[ii])
	}
	return boardExamples, boardLabels, actionsLabels
}

func MatchDecode(dec *gob.Decoder, matchFileIdx int) (match *Match, err error) {
	glog.V(2).Infof("Loading match ...")
	match = &Match{MatchFileIdx: matchFileIdx}
	initial := &Board{}
	initial, match.Actions, match.Scores, err = LoadMatch(dec)
	match.ActionsLabels = make([][]float32, 0, len(match.Actions))
	if err != nil {
		return
	}
	glog.V(2).Infof("Loaded match with %d actions", len(match.Actions))
	initial.BuildDerived()
	match.Boards = make([]*Board, 1, len(match.Actions)+1)
	match.Boards[0] = initial
	board := initial
	for _, action := range match.Actions {
		var actionsLabels []float32
		if !action.IsSkipAction() {
			// When loading a match use one-hot encoding for labels.
			actionIdx := board.FindActionDeep(action)
			actionsLabels = ai.OneHotEncoding(board.NumActions(), actionIdx)
		} else {
			if board.NumActions() != 0 {
				log.Panicf("Unexpected SKIP_ACTION for board with %d actions, MatchFileIdx=%d",
					board.NumActions(), match.MatchFileIdx)
			}
		}
		match.ActionsLabels = append(match.ActionsLabels, actionsLabels)
		board = board.Act(action)
		match.Boards = append(match.Boards, board)
	}
	return
}

var (
	stepUI   = ascii_ui.NewUI(true, false)
	muStepUI sync.Mutex
)

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
	lastWasSkip := false
	for !board.IsFinished() {
		player := board.NextPlayer
		if swapped {
			player = 1 - player
		}
		glog.V(1).Infof("\n\nMatch %d: player %d at turn %d (#actions=%d)\n\n",
			matchNum, player, board.MoveNumber, len(board.Derived.Actions))
		var action Action
		score := float32(0)
		var actionLabels []float32
		if len(board.Derived.Actions) == 0 {
			// Auto-play skip move.
			action = SKIP_ACTION
			board = board.Act(action)
			lastWasSkip = true
			if len(board.Derived.Actions) == 0 {
				log.Panicf("No moves to either side!?\n\n%v\n", board)
			}
		} else {
			action, board, score, actionLabels = reorderedPlayers[board.NextPlayer].Play(board)
			if lastWasSkip {
				// Use inverse of this score for previous "NOOP" move.
				match.Scores[len(match.Scores)-1] = -score
				lastWasSkip = false
			}
		}
		match.Actions = append(match.Actions, action)
		match.Boards = append(match.Boards, board)
		match.Scores = append(match.Scores, score)
		match.ActionsLabels = append(match.ActionsLabels, actionLabels)

		if *flag_printSteps {
			muStepUI.Lock()
			fmt.Printf("Match %d action take: %s\n", match.MatchFileIdx, action)
			stepUI.PrintBoard(board)
			fmt.Println("")
			muStepUI.Unlock()
		}
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

func setAutoBatchSizes(batchSize int) {
	glog.V(1).Infof("setAutoBatchSize(%d), max=%d", batchSize, *flag_maxAutoBatch)
	if *flag_maxAutoBatch > 0 && batchSize > *flag_maxAutoBatch {
		batchSize = *flag_maxAutoBatch
	}
	for _, player := range players {
		if tfscorer, ok := player.Scorer.(*tensorflow.Scorer); ok {
			tfscorer.SetBatchSize(batchSize)
		}
	}
}

// runMatches run --num_matches number of matches, and write the resulting matches
// to the given channel.
func runMatches(results chan<- *Match) {
	if *flag_winsOnly {
		*flag_wins = true
	}
	numMatchesToPlay := *flag_numMatches
	if numMatchesToPlay == 0 {
		numMatchesToPlay = 1
	}
	// Run at most GOMAXPROCS simultaneously.
	var wg sync.WaitGroup
	parallelism := runtime.GOMAXPROCS(0)
	if *flag_parallelism > 0 {
		parallelism = *flag_parallelism
	}
	setAutoBatchSizes(parallelism / 4)
	glog.V(1).Infof("Parallelism for running matches=%d", parallelism)
	semaphore := make(chan bool, parallelism)
	done := false
	wins := 0
	matchCount := 0
	for ; !done; matchCount++ {
		wg.Add(1)
		semaphore <- true
		go func(matchNum int) {
			defer wg.Done()
			match := runMatch(matchNum)
			if !match.FinalBoard().Draw() {
				wins++
				if *flag_wins {
					done = done || (wins >= numMatchesToPlay)
					if !done {
						glog.V(1).Infof("%d matches with wins still needed.", numMatchesToPlay-wins)
					} else {
						glog.V(1).Infof("Got enough wins, just waiting current matches to end.")
					}
				}
			}
			<-semaphore
			if *flag_winsOnly && match.FinalBoard().Draw() {
				return
			}
			results <- match
		}(matchCount)
		if !*flag_wins {
			done = done || (matchCount+1 >= numMatchesToPlay)
		}
		glog.V(1).Infof("Started match %d, done=%v (wins so far=%d)", matchCount, done, wins)
	}

	// Gradually decrease the batching level.
	go func() {
		for ii := parallelism; ii > 0; ii-- {
			semaphore <- true
			setAutoBatchSizes(ii / 4)
		}
	}()

	wg.Wait()
	glog.V(1).Infof("Played %d matches, with %d wins", matchCount, wins)
	close(results)
}

func backupName(filename string) string {
	return filename + "~"
}

func temporaryName(filename string) string {
	return filename + ".tmp"
}

// Open file for writing. If filename already exists rename it by appending an "~" suffix.
func openWriterAndBackup(filename string) io.WriteCloser {
	file, err := os.Create(temporaryName(filename))
	if err != nil {
		log.Panicf("Failed to create temporary save file '%s': %v", temporaryName(filename), err)
	}
	return file
}

func renameToFinal(filename string) {
	if _, err := os.Stat(filename); err == nil {
		err = os.Rename(filename, backupName(filename))
		if err != nil {
			log.Printf("Failed to rename '%s' to '%s': %v", filename, backupName(filename), err)
		}
	}
	err := os.Rename(temporaryName(filename), filename)
	if err != nil {
		log.Printf("Failed to rename '%s' to '%s': %v", temporaryName(filename), filename, err)
	}
}

// Load matches, and automatically build boards.
func loadMatches(results chan<- *Match) {
	filenames, err := filepath.Glob(*flag_loadMatches)
	if err != nil {
		log.Panicf("Invalid pattern '%s' for loading matches: %v", *flag_loadMatches, err)
	}
	if len(filenames) == 0 {
		log.Panicf("Did not find any files matching '%s'", *flag_loadMatches)
	}

	var matchesCount, matchesIdx, numWins int

LoopFilenames:
	for _, filename := range filenames {
		glog.V(1).Infof("Scanning file %s\n", filename)
		file, err := os.Open(filename)
		if err != nil {
			log.Panicf("Cannot open '%s' for reading: %v", filename, err)
		}
		dec := gob.NewDecoder(file)
		for *flag_numMatches == 0 || matchesCount < *flag_numMatches {
			match, err := MatchDecode(dec, matchesIdx)
			matchesIdx++
			if err == io.EOF {
				break
			}
			if err != nil {
				glog.Errorf("Cannot read any more matches in %s: %v", filename, err)
				break
			}
			if *flag_loadOnlyMatch >= 0 {
				// Only take teh specific match.
				if match.MatchFileIdx != *flag_loadOnlyMatch {
					glog.V(1).Infof("Skipping match %d\n", match.MatchFileIdx)
					continue
				} else {
					glog.V(1).Infof("Found match %d\n", *flag_loadOnlyMatch)
					matchesCount++
					if !match.FinalBoard().Draw() {
						numWins++
					}
					results <- match
					break LoopFilenames
				}
			}
			if *flag_winsOnly && match.FinalBoard().Draw() {
				continue
			}
			matchesCount++
			if !match.FinalBoard().Draw() {
				numWins++
			}
			results <- match
		}
	}
	glog.Infof("%d matches loaded (%d wins), %d used", matchesIdx, numWins, matchesCount)
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
		players[ii] = ai_players.NewAIPlayer(*flag_players[ii], *flag_numMatches == 1)
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
	var file io.WriteCloser
	if *flag_saveMatches != "" {
		file = openWriterAndBackup(*flag_saveMatches)
		enc = gob.NewEncoder(file)
	}

	count := 0
	var (
		boardExamples []*Board
		boardLabels   []float32
		actionsLabels [][]float32
	)
	ui := ascii_ui.NewUI(true, false)
	for match := range matches {
		count++
		if enc != nil {
			match.Encode(enc)
		}
		if *flag_train {
			boardExamples, boardLabels, actionsLabels = match.AppendLabeledExamples(
				boardExamples, boardLabels, actionsLabels)
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

	// Finalize file with matches.
	if *flag_saveMatches != "" {
		file.Close()
		renameToFinal(*flag_saveMatches)
	}

	// Train with examples.
	if *flag_train {
		trainFromExamples(boardExamples, boardLabels, actionsLabels)
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
