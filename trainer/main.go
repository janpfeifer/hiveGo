package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sync"

	"github.com/janpfeifer/hiveGo/ai"

	"github.com/golang/glog"
	ai_players "github.com/janpfeifer/hiveGo/ai/players"
	_ "github.com/janpfeifer/hiveGo/ai/search/ab"
	_ "github.com/janpfeifer/hiveGo/ai/search/mcts"
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
		"max_moves", DEFAULT_MAX_MOVES, "Max moves before game is assumed to be a draw.")

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

	flag_lastActions  = flag.Int("last_actions", 0, "If set > 0, on the given number of last moves of each match are used for training.")
	flag_startActions = flag.Int("start_actions", -1,
		"Must be used in combination with --last_actions. If set, it defines the first action to start using "+
			"for training, and --last_actions will define how many actions to learn from.")

	flag_train           = flag.Bool("train", false, "Set to true to train with match data.")
	flag_trainLoops      = flag.Int("train_loops", 1, "After acquiring data for all matches, how many times to loop the training over it.")
	flag_trainValidation = flag.Int("train_validation", 0, "Percentage (int value) of matches used for validation (evaluation only).")
	flag_learningRate    = flag.Float64("learning_rate", 1e-5, "Learning rate when learning")
	flag_rescore         = flag.Bool("rescore", false, "If to rescore matches.")
	flag_distill         = flag.Bool("distill", false,
		"If set it will simply distill from --ai1 to --ai0, without serching for best moves.")

	flag_parallelism = flag.Int("parallelism", 0, "If > 0 ignore GOMAXPROCS and play "+
		"these many matches simultaneously.")
	flag_maxAutoBatch = flag.Int("max_auto_batch", 0, "If > 0 ignore at most do given value of "+
		"auto-batch for tensorflow evaluations.")

	flag_continuosRescoreAndTrain = flag.Bool("rescore_and_train", false, "If set, continuously rescore and train matches.")
	flag_rescoreAndTrainPoolSize  = flag.Int("rescore_and_train_pool_size", 10000,
		"How many board positions to keep in pool (in a rotating buffer) used to train. ")
	flag_rescoreAndTrainIssueLearn = flag.Int("rescore_and_train_issue_learn", 10,
		"After how many rescored matches/action to issue another learning mini-batch.")

	// AI for the players. If their configuration is exactly the same, they will point to the same object.
	players = [2]*ai_players.SearcherScorerPlayer{nil, nil}
)

func init() {
	flag.BoolVar(&tensorflow.CpuOnly, "cpu", false, "Force to use CPU, even if GPU is available")
}

// Results and if the players were swapped.
type Match struct {
	mu sync.Mutex

	// Match number: only for printing.
	MatchNum int

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
	if err := SaveMatch(enc, m.Boards[0].MaxMoves, m.Actions, m.Scores, m.ActionsLabels); err != nil {
		log.Panicf("Failed to encode match: %v", err)
	}
}

// SelectRangeOfActions returns range of actions to be used, and the amount of samples
// to use when training.
func (m *Match) SelectRangeOfActions() (from, to int, samplesPerAction []int) {
	from = 0
	to = len(m.Actions)
	if *flag_lastActions > 0 && *flag_lastActions < len(m.Actions) {
		from = len(m.Actions) - *flag_lastActions
	}
	if *flag_startActions >= 0 {
		from = *flag_startActions
		if from > len(m.Actions) {
			return -1, -1, nil
		}
		if *flag_lastActions > 0 && from+*flag_lastActions < to {
			to = from + *flag_lastActions
		}
	}

	// Sampling strategy:
	count := to - from
	samplesPerAction = make([]int, count)
	multiplier := 1
	if count > 50 {
		multiplier = 2
	}
	for idx := from; idx < to; idx++ {
		actionsToEnd := len(m.Actions) - idx - 1
		samples := 1
		if actionsToEnd < 10 {
			samples += multiplier
		}
		if actionsToEnd < 6 {
			samples += multiplier
		}
		if actionsToEnd < 3 {
			samples += multiplier
		}
		samplesPerAction[idx-from] = samples
	}
	return
}

type LabeledExamples struct {
	boardExamples []*Board
	boardLabels   []float32
	actionsLabels [][]float32
}

func (le *LabeledExamples) Len() int {
	return len(le.boardExamples)
}

// AppendLabeledExamples will add examples for learning _for Player 0 only_.
func (m *Match) AppendLabeledExamplesForPlayers(le *LabeledExamples, includedPlayers [2]bool) {
	from, to, samples := m.SelectRangeOfActions()
	if to == -1 {
		// No actions selected.
		return
	}
	glog.V(2).Infof("Making LabeledExample, version=%d, included players %v",
		players[0].Scorer.Version(), includedPlayers)
	for ii := from; ii < to; ii++ {
		if includedPlayers[m.Boards[ii].NextPlayer] && !m.Boards[ii].IsFinished() &&
			m.Boards[ii].NumActions() > 1 {
			for jj := 0; jj < samples[ii-from]; jj++ {
				le.boardExamples = append(le.boardExamples, m.Boards[ii])
				le.boardLabels = append(le.boardLabels, m.Scores[ii])
				le.actionsLabels = append(le.actionsLabels, m.ActionsLabels[ii])
			}
		}
	}
}

func (m *Match) AppendLabeledExamples(le *LabeledExamples) {
	m.AppendLabeledExamplesForPlayers(le, [2]bool{true, true})
}

// playActions fills the boards by playing one action at a time. The
// initial board given must contain MaxMoves set.
func (match *Match) playActions(initial *Board) {
	initial.BuildDerived()
	match.Boards = make([]*Board, 1, len(match.Actions)+1)
	match.Boards[0] = initial
	board := initial
	for _, action := range match.Actions {
		board = board.Act(action)
		match.Boards = append(match.Boards, board)
	}
}

// fillActionLabels will fill the ActionLabels attribute
// with the one-hot-encoding of the action actually played.
func (match *Match) fillActionLabelsWithActionTaken() {
	match.ActionsLabels = make([][]float32, 0, len(match.Actions))
	for moveIdx, actionTaken := range match.Actions {
		board := match.Boards[moveIdx]
		var actionsLabels []float32
		if !actionTaken.IsSkipAction() {
			// When loading a match use one-hot encoding for labels.
			actionIdx := board.FindActionDeep(actionTaken)
			actionsLabels = ai.OneHotEncoding(board.NumActions(), actionIdx)
		} else {
			if board.NumActions() != 0 {
				log.Panicf("Unexpected SKIP_ACTION for board with %d actions, MatchFileIdx=%d",
					board.NumActions(), match.MatchFileIdx)
			}
		}
		match.ActionsLabels = append(match.ActionsLabels, actionsLabels)
	}
}

func MatchDecode(dec *gob.Decoder, matchFileIdx int) (match *Match, err error) {
	glog.V(2).Infof("Loading match ...")
	match = &Match{MatchFileIdx: matchFileIdx}
	var initial *Board
	initial, match.Actions, match.Scores, match.ActionsLabels, err = LoadMatch(dec)
	if err != nil {
		return
	}
	match.playActions(initial)
	if match.ActionsLabels == nil {
		match.fillActionLabelsWithActionTaken()
	}
	glog.V(2).Infof("Loaded match with %d actions", len(match.Actions))
	return
}

var (
	stepUI   = ascii_ui.NewUI(true, false)
	muStepUI sync.Mutex
)

func runMatch(matchNum int) *Match {
	swapped := (matchNum%2 != 0)
	matchName := fmt.Sprintf("%d (swap=%v)", matchNum, swapped)
	board := NewBoard()
	board.MaxMoves = *flag_maxMoves
	match := &Match{MatchNum: matchNum, Swapped: swapped, Boards: []*Board{board}}
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
			action, board, score, actionLabels = reorderedPlayers[board.NextPlayer].Play(board,
				matchName)
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
		started := 0
		if match.Swapped {
			started = 1
		}
		glog.V(1).Infof("\n\nMatch %d: finished at turn %d, %s (Player %d started)\n\n",
			matchNum, match.FinalBoard().MoveNumber, msg, started)
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

var (
	// Set to true if same AI is playing both sides.
	isSamePlayer = false
)

func setAutoBatchSizesForParallelism(parallelism int) {
	autoBatchSize := parallelism / 4
	if isSamePlayer {
		autoBatchSize = parallelism / 2
	}
	setAutoBatchSizes(autoBatchSize)
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
	parallelism := getParallelism()
	setAutoBatchSizesForParallelism(parallelism)
	glog.V(1).Infof("Parallelism for running matches=%d", parallelism)
	semaphore := make(chan bool, parallelism)
	done := false
	wins := 0
	matchCount := 0
	doneCount := 0
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
			doneCount++
			glog.V(1).Infof("Match %d finished: %d matches done, %d wins (non-draws)", matchNum, doneCount, wins)
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
			setAutoBatchSizesForParallelism(ii)
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

// Save all given matches to location given --save_matches.
// It creates file into a temporary file first, and move
// to final destination once finished.
func saveMatches(matches []*Match) {
	file := openWriterAndBackup(*flag_saveMatches)
	enc := gob.NewEncoder(file)
	for _, match := range matches {
		match.Encode(enc)
	}
	file.Close()
	renameToFinal(*flag_saveMatches)
}

// Report on matches as they are being played/generated.
func reportMatches(results <-chan *Match) (matches []*Match) {
	// Read results.
	totalWins := [3]int{0, 0, 0}
	totalMoves := 0
	ui := ascii_ui.NewUI(true, false)
	for match := range results {
		matches = append(matches, match)

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

	// Print totals.
	count := len(matches)
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

	return
}

// getParallelism returns the parallelism.
func getParallelism() (parallelism int) {
	parallelism = runtime.GOMAXPROCS(0)
	if *flag_parallelism > 0 {
		parallelism = *flag_parallelism
	}
	return
}

// main orchestrates playing, loading, rescoring, saving and training of matches.
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

	if *flag_rescore && !*flag_train && *flag_saveMatches == "" {
		log.Fatal("Flag --rescore set, but not --train and not --save_matches. Not sure what to do.")
	}
	if *flag_maxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flag_maxMoves)
	}

	// Create AI players. If they are the same, reuse -- sharing same TF Session can be more
	// efficient.
	glog.Infof("Creating player 0 from '%s'", *flag_players[0])
	players[0] = ai_players.NewAIPlayer(*flag_players[0], *flag_numMatches == 1)
	if *flag_players[1] == *flag_players[0] {
		players[1] = players[0]
		isSamePlayer = true
		glog.Infof("Player 1 is the same as player 0, reusing AI player object.")
	} else {
		glog.Infof("Creating player 1 from '%s'", *flag_players[1])
		players[1] = ai_players.NewAIPlayer(*flag_players[1], *flag_numMatches == 1)
	}

	if *flag_continuosPlayAndTrain {
		playAndTrain()
		return
	}

	// Run/load matches.
	results := make(chan *Match)
	if *flag_loadMatches != "" {
		go loadMatches(results)
	} else {
		go runMatches(results)
	}
	if *flag_rescore {
		rescored := make(chan *Match)
		go rescoreMatches(results, rescored)
		results = rescored
	}

	// Collect resulting matches, optionally reporting on them.
	var matches []*Match
	if !*flag_rescore {
		matches = reportMatches(results)
	} else {
		for match := range results {
			matches = append(matches, match)
		}
	}

	if *flag_saveMatches != "" {
		saveMatches(matches)
	}
	if *flag_train {
		trainFromMatches(matches)
	}
	if *flag_continuosRescoreAndTrain {
		rescoreAndTrain(matches)
	}
}
