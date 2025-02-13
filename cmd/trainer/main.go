package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/internal/ai/tensorflow"
	players2 "github.com/janpfeifer/hiveGo/internal/players"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"github.com/pkg/errors"
	"io"
	"k8s.io/klog/v2"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sync"

	"github.com/janpfeifer/hiveGo/ai/tfddqn"
	"github.com/janpfeifer/hiveGo/internal/search/ab"
	"github.com/janpfeifer/hiveGo/internal/search/mcts"
)

var _ = fmt.Printf

var (
	flagCpuProfile = flag.String("cpu_profile", "", "write cpu profile to `file`")

	flagPlayers = [2]*string{
		flag.String("ai0", "", "Configuration string for ai playing as the starting player."),
		flag.String("ai1", "", "Configuration string for ai playing as the second player."),
	}

	flagMaxMoves = flag.Int(
		"max_moves", DefaultMaxMoves, "Max moves before game is assumed to be a draw.")

	flagNumMatches = flag.Int("num_matches", 0, "Number of matches to play. If larger "+
		"than one, starting position is alternated. Value of 0 means 1 match to play, or load all file.")
	flagPrint       = flag.Bool("print", false, "Print board at the end of the match.")
	flagPrintSteps  = flag.Bool("print_steps", false, "Print board at each step.")
	flagSaveMatches = flag.String("save_matches", "", "File name where to save matches.")
	flagLoadMatches = flag.String("load_matches", "",
		"Instead of actually playing matches, load pre-generated ones.")
	flagOnlyLoadMatch = flag.Int("match_idx", -1, "If set it will only load this one "+
		"specific match -- useful for debugging.")
	flagWins     = flag.Bool("wins", false, "Counts only matches with wins.")
	flagWinsOnly = flag.Bool("wins_only", false, "Counts only matches with wins (like --wins) and discards draws.")

	flagLastActions  = flag.Int("last_actions", 0, "If set > 0, on the given number of last moves of each match are used for training.")
	flagStartActions = flag.Int("start_actions", -1,
		"Must be used in combination with --last_actions. If set, it defines the first action to start using "+
			"for training, and --last_actions will define how many actions to learn from.")

	flagTrain           = flag.Bool("train", false, "Set to true to train with match data.")
	flagTrainLoops      = flag.Int("train_loops", 1, "After acquiring data for all matches, how many times to loop the training over it.")
	flagTrainValidation = flag.Int("train_validation", 0, "Percentage (int value) of matches used for validation (evaluation only).")
	flagLearningRate    = flag.Float64("learning_rate", 1e-5, "Learning rate when learning")
	flagRescore         = flag.Bool("rescore", false, "If to rescore matches.")
	flagDistill         = flag.Bool("distill", false,
		"If set it will simply distill from --ai1 to --ai0, without searching for best moves.")
	flagLearnWithEndScore = flag.Bool("learn_with_end_score",
		true, "If true will use the final score to learn.")

	flagParallelism = flag.Int("parallelism", 0, "If > 0 ignore GOMAXPROCS and play "+
		"these many matches simultaneously.")
	flagMaxAutoBatch = flag.Int("max_auto_batch", 0, "If > 0 ignore at most do given value of "+
		"auto-batch for tensorflow evaluations.")

	flagContinuosRescoreAndTrain = flag.Bool("rescore_and_train", false, "If set, continuously rescore and train matches.")
	flagRescoreAndTrainPoolSize  = flag.Int("rescore_and_train_pool_size", 10000,
		"How many board positions to keep in pool (in a rotating buffer) used to train. ")
	flagRescoreAndTrainIssueLearn = flag.Int("rescore_and_train_issue_learn", 10,
		"After how many rescored matches/action to issue another learning mini-batch.")

	// AI for the players. If their configuration is exactly the same, they will point to the same object.
	players = [2]*players2.SearcherScorer{nil, nil}
)

func init() {
	flag.BoolVar(&tensorflow.CpuOnly, "cpu", false, "Force to use CPU, even if GPU is available")
}

// Match holds the results and whether the players were swapped.
// The first dimension of all slices is the move (ply) number.
type Match struct {
	mu sync.Mutex

	// Match number: only for printing.
	MatchNum int

	// Whether p0/p1 swapped positions in this match.
	Swapped bool

	// Match actions, alternating players.
	Actions []Action

	// Probability distributions over actions, to learn from. Only set for MCTS.
	// For AlphaBetaPruning these are left nil.
	ActionsLabels [][]float32

	// All board states of the game: 1 more than the number of actions.
	Boards []*Board

	// Scores for each board position. Can either be calculated during
	// the match, or re-generated when re-loading a match.
	Scores []float32

	// Index of the match in the file it was loaded from.
	MatchFileIdx int
}

func (m *Match) FinalBoard() *Board { return m.Boards[len(m.Boards)-1] }

func (m *Match) Encode(enc *gob.Encoder) error {
	return SaveMatch(enc, m.Boards[0].MaxMoves, m.Actions, m.Scores, m.ActionsLabels)
}

// SelectRangeOfActions returns range of actions to be used, and the amount of samples
// to use when training.
func (m *Match) SelectRangeOfActions() (from, to int, samplesPerAction []int) {
	from = 0
	to = len(m.Actions)
	if *flagLastActions > 0 && *flagLastActions < len(m.Actions) {
		from = len(m.Actions) - *flagLastActions
	}
	if *flagStartActions >= 0 {
		from = *flagStartActions
		if from > len(m.Actions) {
			return -1, -1, nil
		}
		if *flagLastActions > 0 && from+*flagLastActions < to {
			to = from + *flagLastActions
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

// AppendLabeledExamplesForPlayers adds examples for learning for the selected player(s).
func (m *Match) AppendLabeledExamplesForPlayers(le *LabeledExamples, includedPlayers [2]bool) {
	from, to, samples := m.SelectRangeOfActions()
	if to == -1 {
		// No actions selected.
		return
	}
	klog.V(2).Infof("Making LabeledExample, version=%d, included players %v",
		players[0].Scorer.Version(), includedPlayers)
	for ii := from; ii < to; ii++ {
		if includedPlayers[m.Boards[ii].NextPlayer] && !m.Boards[ii].IsFinished() &&
			m.Boards[ii].NumActions() > 1 {

			if klog.V(3).Enabled() {
				fmt.Println("")
				stepUI.PrintBoard(m.Boards[ii])
				score, actionsPred := players[0].Scorer.Score(m.Boards[ii], true)
				fmt.Printf("Score: %g, ActionsPred: %v\n\n", score, actionsPred)
			}

			for jj := 0; jj < samples[ii-from]; jj++ {
				klog.V(2).Infof("Learning board scores: %v", m.Scores[ii])
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
func (m *Match) playActions(initial *Board) {
	initial.BuildDerived()
	m.Boards = make([]*Board, 1, len(m.Actions)+1)
	m.Boards[0] = initial
	board := initial
	for _, action := range m.Actions {
		board = board.Act(action)
		m.Boards = append(m.Boards, board)
	}
}

// fillActionLabels will fill the ActionLabels attribute
// with the one-hot-encoding of the action actually played.
func (m *Match) fillActionLabelsWithActionTaken() {
	m.ActionsLabels = make([][]float32, 0, len(m.Actions))
	for moveIdx, actionTaken := range m.Actions {
		board := m.Boards[moveIdx]
		var actionsLabels []float32
		if !actionTaken.IsSkipAction() {
			// When loading a match use one-hot encoding for labels.
			actionIdx := board.FindActionDeep(actionTaken)
			actionsLabels = ai.OneHotEncoding(board.NumActions(), actionIdx)
		} else {
			if board.NumActions() != 0 {
				log.Panicf("Unexpected SkipAction for board with %d actions, MatchFileIdx=%d",
					board.NumActions(), m.MatchFileIdx)
			}
		}
		m.ActionsLabels = append(m.ActionsLabels, actionsLabels)
	}
}

func MatchDecode(dec *gob.Decoder, matchFileIdx int) (match *Match, err error) {
	klog.V(2).Infof("Loading match ...")
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
	klog.V(2).Infof("Loaded match with %d actions", len(match.Actions))
	return
}

var (
	stepUI   = cli.New(true, false)
	muStepUI sync.Mutex
)

func runMatch(matchNum int) *Match {
	swapped := (matchNum%2 != 0)
	matchName := fmt.Sprintf("%d (swap=%v)", matchNum, swapped)
	board := NewBoard()
	board.MaxMoves = *flagMaxMoves
	match := &Match{MatchNum: matchNum, Swapped: swapped, Boards: []*Board{board}}

	// Run match.
	lastWasSkip := false
	for !board.IsFinished() {
		player := board.NextPlayer
		if swapped {
			player = 1 - player
		}
		klog.V(1).Infof(
			"\n\nMatch %d: player %d at turn %d (#valid actions=%d)\n\n",
			matchNum, player, board.MoveNumber, len(board.Derived.Actions))
		var action Action
		score := float32(0)
		var actionLabels []float32
		if len(board.Derived.Actions) == 0 {
			// Auto-play skip move.
			action = SkipAction
			board = board.Act(action)
			lastWasSkip = true
			if len(board.Derived.Actions) == 0 {
				log.Panicf("No moves to either side!?\n\n%v\n", board)
			}
		} else {
			action, board, score, actionLabels = players[player].Play(board,
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

		if *flagPrintSteps {
			muStepUI.Lock()
			fmt.Printf("Match %d action taken: %s\n", match.MatchFileIdx, action)
			stepUI.PrintBoard(board)
			fmt.Println("")
			muStepUI.Unlock()
		}
	}

	finalBoard := match.FinalBoard()
	if !finalBoard.IsFinished() {
		log.Panic("Match %d stopped before being finished!?", matchNum)
	}

	if klog.V(1).Enabled() {
		var msg string
		if finalBoard.Draw() {
			if finalBoard.MoveNumber > finalBoard.MaxMoves {
				msg = "match was a draw (MaxMoves reached)!"
			} else {
				msg = fmt.Sprintf("match was a draw (%d repeats)!",
					finalBoard.Derived.Repeats)
			}
		} else {
			player := finalBoard.Winner()
			if swapped {
				player = 1 - player
			}
			msg = fmt.Sprintf("player %d won!", player)
		}
		started := 0
		if match.Swapped {
			started = 1
		}
		klog.V(1).Infof("\n\nMatch %d: finished at turn %d, %s (Player %d started)\n\n",
			matchNum, finalBoard.MoveNumber, msg, started)
	}

	return match
}

func setAutoBatchSizes(batchSize int) {
	klog.V(1).Infof("setAutoBatchSize(%d), max=%d", batchSize, *flagMaxAutoBatch)
	if *flagMaxAutoBatch > 0 && batchSize > *flagMaxAutoBatch {
		batchSize = *flagMaxAutoBatch
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
	if *flagWinsOnly {
		*flagWins = true
	}
	numMatchesToPlay := *flagNumMatches
	if numMatchesToPlay == 0 {
		numMatchesToPlay = 1
	}
	// Run at most GOMAXPROCS simultaneously.
	var wg sync.WaitGroup
	parallelism := getParallelism()
	setAutoBatchSizesForParallelism(parallelism)
	klog.V(1).Infof("Parallelism for running matches=%d", parallelism)
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
				if *flagWins {
					done = done || (wins >= numMatchesToPlay)
					if !done {
						klog.V(1).Infof("%d matches with wins still needed.", numMatchesToPlay-wins)
					} else {
						klog.V(1).Infof("Got enough wins, just waiting current matches to end.")
					}
				}
			}
			doneCount++
			klog.V(1).Infof("Match %d finished: %d matches done, %d wins (non-draws)", matchNum, doneCount, wins)
			<-semaphore
			if *flagWinsOnly && match.FinalBoard().Draw() {
				return
			}
			results <- match
		}(matchCount)
		if !*flagWins {
			done = done || (matchCount+1 >= numMatchesToPlay)
		}
		klog.V(1).Infof("Started match %d, done=%v (wins so far=%d)", matchCount, done, wins)
	}

	// Gradually decrease the batching level.
	go func() {
		for ii := parallelism; ii > 0; ii-- {
			semaphore <- true
			setAutoBatchSizesForParallelism(ii)
		}
	}()

	wg.Wait()
	klog.V(1).Infof("Played %d matches, with %d wins", matchCount, wins)
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
		log.Panicf("Failed to create temporary save file %q: %v", temporaryName(filename), err)
	}
	return file
}

func renameToFinal(filename string) {
	if _, err := os.Stat(filename); err == nil {
		err = os.Rename(filename, backupName(filename))
		if err != nil {
			log.Printf("Failed to rename %q to %q: %v", filename, backupName(filename), err)
		}
	}
	err := os.Rename(temporaryName(filename), filename)
	if err != nil {
		log.Printf("Failed to rename %q to %q: %v", temporaryName(filename), filename, err)
	}
}

// Load matches, and automatically build boards.
func loadMatches(results chan<- *Match) {
	filenames, err := filepath.Glob(*flagLoadMatches)
	if err != nil {
		log.Panicf("Invalid pattern %q for loading matches: %v", *flagLoadMatches, err)
	}
	if len(filenames) == 0 {
		log.Panicf("Did not find any files matching %q", *flagLoadMatches)
	}

	var matchesCount, matchesIdx, numWins int

LoopFilenames:
	for _, filename := range filenames {
		klog.V(1).Infof("Scanning file %s\n", filename)
		file, err := os.Open(filename)
		if err != nil {
			log.Panicf("Cannot open %q for reading: %v", filename, err)
		}
		dec := gob.NewDecoder(file)
		for *flagNumMatches == 0 || matchesCount < *flagNumMatches {
			match, err := MatchDecode(dec, matchesIdx)
			matchesIdx++
			if err == io.EOF {
				break
			}
			if err != nil {
				klog.Errorf("Cannot read any more matches in %s: %v", filename, err)
				break
			}
			if *flagOnlyLoadMatch >= 0 {
				// Only take teh specific match.
				if match.MatchFileIdx != *flagOnlyLoadMatch {
					klog.V(1).Infof("Skipping match %d\n", match.MatchFileIdx)
					continue
				} else {
					klog.V(1).Infof("Found match %d\n", *flagOnlyLoadMatch)
					matchesCount++
					if !match.FinalBoard().Draw() {
						numWins++
					}
					results <- match
					break LoopFilenames
				}
			}
			if *flagWinsOnly && match.FinalBoard().Draw() {
				continue
			}
			matchesCount++
			if !match.FinalBoard().Draw() {
				numWins++
			}
			results <- match
		}
	}
	klog.Infof("%d matches loaded (%d wins), %d used", matchesIdx, numWins, matchesCount)
	close(results)
}

// Save all given matches to location given --save_matches.
// It creates file into a temporary file first, and move
// to final destination once finished.
func saveMatches(matches []*Match) error {
	file := openWriterAndBackup(*flagSaveMatches)
	enc := gob.NewEncoder(file)
	for _, match := range matches {
		match.Encode(enc)
	}
	err := file.Close()
	if err != nil {
		return errors.Wrapf(err, "failed to close saved matches in %q", *flagSaveMatches)
	}
	renameToFinal(*flagSaveMatches)
}

// Report on matches as they are being played/generated.
func reportMatches(results <-chan *Match) (matches []*Match) {
	// Read results.
	totalWins := [3]int{0, 0, 0}
	totalMoves := 0
	ui := cli.New(true, false)
	for match := range results {
		matches = append(matches, match)

		// Accounting.
		board := match.FinalBoard()
		wins := board.Derived.Wins
		if match.Swapped {
			wins[0], wins[1] = wins[1], wins[0]
		}
		if *flagPrint {
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
	if *flagParallelism > 0 {
		parallelism = *flagParallelism
	}
	return
}

// main orchestrates playing, loading, rescoring, saving and training of matches.
func main() {
	flag.Parse()
	if *flagCpuProfile != "" {
		f, err := os.Create(*flagCpuProfile)
		if err != nil {
			klog.Fatal("could not create CPU profile: ", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			klog.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *flagMaxMoves <= 0 {
		log.Fatalf("Invalid --max_moves=%d", *flagMaxMoves)
	}

	// Create AI players. If they are the same, reuse -- sharing same TF Session can be more
	// efficient.
	klog.Infof("Creating player 0 from %q", *flagPlayers[0])
	players[0] = players2.New(*flagPlayers[0], *flagNumMatches == 1)
	if *flagPlayers[1] == *flagPlayers[0] {
		players[1] = players[0]
		isSamePlayer = true
		klog.Infof("Player 1 is the same as player 0, reusing AI player object.")
	} else {
		klog.Infof("Creating player 1 from %q", *flagPlayers[1])
		players[1] = players2.New(*flagPlayers[1], *flagNumMatches == 1)
	}

	if *flag_continuosPlayAndTrain {
		playAndTrain()
		return
	}

	// Run/load matches.
	results := make(chan *Match)
	if *flagLoadMatches != "" {
		go loadMatches(results)
	} else {
		go runMatches(results)
	}
	if *flagRescore {
		rescored := make(chan *Match)
		go rescoreMatches(results, rescored)
		results = rescored
	}

	// Collect resulting matches, optionally reporting on them.
	var matches []*Match
	if !*flagRescore {
		matches = reportMatches(results)
	} else {
		for match := range results {
			matches = append(matches, match)
		}
	}

	if *flagSaveMatches != "" {
		saveMatches(matches)
	}
	if *flagTrain {
		trainFromMatches(matches)
	}
	if *flagContinuosRescoreAndTrain {
		rescoreAndTrain(matches)
	}
}
