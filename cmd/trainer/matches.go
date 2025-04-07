package main

import (
	"context"
	"encoding/gob"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"github.com/janpfeifer/hiveGo/internal/ui/spinning"
	"github.com/pkg/errors"
	"io"
	"k8s.io/klog/v2"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
)

// Match holds the results and whether the players were swapped.
// The first dimension of all slices is the move (ply) number.
type Match struct {
	mu sync.Mutex

	// Match number: only for printing.
	MatchNum int

	// Which AI played this match as the first/last player.
	PlayersIdx [2]int

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

// FinalBoard position of a match.
func (m *Match) FinalBoard() *Board {
	if len(m.Boards) == 0 {
		return nil
	}
	return m.Boards[len(m.Boards)-1]
}

// Encode the match (save it) into the given encoder.
func (m *Match) Encode(enc Encoder) error {
	return EncodeMatch(enc, m.Boards[0].MaxMoves, m.Actions, m.Scores, m.ActionsLabels)
}

// SelectRangeOfActions returns the range (in terms of move numbers in the match) of actions to be used,
// and the amount of samples to use when training, based on the flags: *flagLastActions and *flagStartActions.
//
// So from=0 to=5 means it will use only the first 5 moves of the match.
//
// If *flagLastActions is true, it is taken from the end of the match.
func (m *Match) SelectRangeOfActions() (from, to int, samplesPerAction []int) {
	from = 0
	to = len(m.Actions)
	if *flagStartActions >= 0 {
		from = *flagStartActions
		if from > len(m.Actions) {
			return -1, -1, nil
		}
		if *flagLastActions > 0 {
			from = max(from, len(m.Actions)-*flagLastActions)
		}
	} else if *flagLastActions > 0 {
		from = max(len(m.Actions)-*flagLastActions, 0)
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

// AppendToLabeledBoardsForPlayers appends this match board positions for the selected player(s) to
// the labeledBoards container.
//
// After it reaches the MaxSize new boards appended start rotating the position (replacing older ones).
func (m *Match) AppendToLabeledBoardsForPlayers(labeledBoards *LabeledBoards, includedPlayers [2]bool) {
	from, to, samples := m.SelectRangeOfActions()
	if to == -1 {
		// No actions selected.
		return
	}
	klog.V(2).Infof("Making LabeledExample, scorer=%s, included players %v",
		aiPlayers[0].Scorer, includedPlayers)
	for ii := from; ii < to; ii++ {
		if includedPlayers[m.Boards[ii].NextPlayer] && !m.Boards[ii].IsFinished() &&
			m.Boards[ii].NumActions() > 1 {

			if klog.V(3).Enabled() {
				fmt.Println("")
				stepUI.PrintBoard(m.Boards[ii])
				score := aiPlayers[0].Scorer.BoardScore(m.Boards[ii])
				fmt.Printf("Score: %g\n\n", score)
			}

			for jj := 0; jj < samples[ii-from]; jj++ {
				klog.V(2).Infof("Learning board scores: %v", m.Scores[ii])
				labeledBoards.AddBoard(m.Boards[ii], m.Scores[ii], m.ActionsLabels[ii])
			}
		}
	}
}

// AppendToLabeledBoards appends this match to the given labeledBoards container.
//
// This is a shortcut to AppendToLabeledBoardsForPlayers for both players.
//
// After labeledBoards reaches the MaxSize new boards appended start rotating the position (replacing older ones).
func (m *Match) AppendToLabeledBoards(labeledBoards *LabeledBoards) {
	m.AppendToLabeledBoardsForPlayers(labeledBoards, [2]bool{true, true})
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

// fillActionLabels will fill the ActionsLabels attribute
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

// runMatch from start to end, and return the collected moves and scores.
// It returns nil is the ctx was cancelled at any point.
func runMatch(ctx context.Context, matchNum int, useRandomPlayers bool) *Match {
	if klog.V(1).Enabled() {
		klog.Infof("Starting match %d", matchNum)
		defer klog.Infof("Finished match %d", matchNum)
	}
	if len(aiPlayers) < 0 {
		klog.Fatalf("No AI players configured, cannot run matches")
	}

	board := NewBoard()
	board.MaxMoves = *flagMaxMoves
	match := &Match{MatchNum: matchNum, Boards: []*Board{board}}
	if useRandomPlayers {
		for ii := range 2 {
			match.PlayersIdx[ii] = rand.Intn(len(aiPlayers))
		}
	} else {
		if len(aiPlayers) > 2 {
			klog.Fatalf("Expected 1 or 2 AI players configured, got %d", len(aiPlayers))
		}
		if len(aiPlayers) == 2 {
			// Alternate who plays first.
			if matchNum%2 == 0 {
				match.PlayersIdx[0], match.PlayersIdx[1] = 0, 1
			} else {
				match.PlayersIdx[0], match.PlayersIdx[1] = 1, 0
			}
		}
	}
	matchName := fmt.Sprintf("Match-%05d (First players=%d, Second player=%d)",
		matchNum, match.PlayersIdx[0], match.PlayersIdx[1])

	// Run match.
	lastWasSkip := false
	for !board.IsFinished() {
		prevBoard := board
		if ctx.Err() != nil {
			return nil
		}
		playerNum := board.NextPlayer
		aiPlayer := aiPlayers[match.PlayersIdx[playerNum]]
		if klog.V(1).Enabled() {
			klog.Infof(
				"\n\n%s: %s at turn %d (#valid actions=%d)\n\n",
				matchName, playerNum, board.MoveNumber, len(board.Derived.Actions))
		}
		var action Action
		score := float32(0)
		var actionLabels []float32
		if len(board.Derived.Actions) == 0 {
			// Auto-play skip move.
			action = SkipAction
			board = board.Act(action)
			lastWasSkip = true
			if len(board.Derived.Actions) == 0 {
				klog.Fatalf("No moves to either side!?\n\n%v\n", board)
			}
		} else {
			action, board, score, actionLabels = aiPlayer.Play(board)
			if lastWasSkip {
				// Use inverse of this score for previous "NOOP" move.
				match.Scores[len(match.Scores)-1] = -score
				lastWasSkip = false
			}
		}
		match.Actions = append(match.Actions, action)
		match.Boards = append(match.Boards, board)
		prevBoard.ClearNextBoardsCache() // De-reference any trailing search tree boards from the previous state, GC takes over after that.
		match.Scores = append(match.Scores, score)
		match.ActionsLabels = append(match.ActionsLabels, actionLabels)

		if *flagPrintSteps {
			muStepUI.Lock()
			fmt.Printf("%s action taken: %s\n", matchName, action)
			stepUI.PrintBoard(board)
			fmt.Println("")
			muStepUI.Unlock()
		}
	}
	board.ClearNextBoardsCache() // Erase any trailing search tree.
	if klog.V(1).Enabled() {
		var msg string
		if board.Draw() {
			if board.MoveNumber > board.MaxMoves {
				msg = "match was a draw (MaxMoves reached)!"
			} else {
				msg = fmt.Sprintf("match was a draw (%d repeats)!",
					board.Derived.Repeats)
			}
		} else {
			player := board.Winner()
			winnerAI := aiPlayers[match.PlayersIdx[player]]
			msg = fmt.Sprintf("%s won! (%s)", player, winnerAI)
		}
		klog.Infof("\n\n%s: finished at turn %d, %s\n\n",
			matchNum, board.MoveNumber, msg)
	}
	return match
}

func setAutoBatchSizes(batchSize int) {
	klog.V(1).Infof("setAutoBatchSize(%d), max=%d", batchSize, *flagMaxAutoBatch)
	if *flagMaxAutoBatch > 0 && batchSize > *flagMaxAutoBatch {
		batchSize = *flagMaxAutoBatch
	}
	/*
		for _, player := range aiPlayers {
			if tfscorer, ok := player.Scorer.(*tensorflow.Scorer); ok {
				tfscorer.SetBatchSize(batchSize)
			}
		}
	*/
}

func setAutoBatchSizesForParallelism(parallelism int) {
	autoBatchSize := parallelism / 4
	setAutoBatchSizes(autoBatchSize)
}

// runMatches run --num_matches number of matches, and write the resulting matches
// to the given channel.
func runMatches(ctx context.Context, matchesChan chan<- *Match) {
	// Makes sure we close matchesChan at the end.
	var matchCount, wins int
	defer func() {
		klog.V(1).Infof("Played %d matches, with %d wins", matchCount, wins)
		close(matchesChan)
	}()

	// Process flags.
	if *flagWinsOnly {
		*flagWins = true
	}
	numMatchesToPlay := *flagNumMatches
	if numMatchesToPlay == 0 {
		numMatchesToPlay = 1
	}

	// Run at most GOMAXPROCS (or --parallelism if set) simultaneously.
	var wg sync.WaitGroup
	parallelism := getParallelism()
	klog.V(1).Infof("Parallelism for running matches=%d", parallelism)
	semaphore := make(chan bool, parallelism)
	done := false
	doneCount := 0

	// Start spinner and feedback about matches being played:
	if *flagWins {
		fmt.Printf("Running %d matches with wins ... ", numMatchesToPlay)
	} else {
		fmt.Printf("Running %d matches ... ", numMatchesToPlay)
	}
	spinner := spinning.New(globalCtx)
	defer spinner.Done()

	// Loop scheduling matches to play concurrently.
	for ; !done; matchCount++ {
		select {
		case <-ctx.Done():
			klog.V(1).Infof("Context cancelled, not starting any new matches: %v", ctx.Err())
			break
		case semaphore <- true:
			// Start a new match.
		}
		wg.Add(1)
		go func(matchNum int) {
			defer wg.Done()
			match := runMatch(ctx, matchNum, false) // Not random players, but rather among 2 players (each start half the matches).
			if ctx.Err() != nil {
				klog.Infof("Match %d interrupted: context cancelled with %v", matchNum, ctx.Err())
			}
			if match == nil {
				// Context interrupted, quick.
				return
			}
			if match.FinalBoard() != nil && !match.FinalBoard().Draw() {
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

			// Allow next match to start already.
			<-semaphore
			if *flagWinsOnly && match.FinalBoard().Draw() {
				return
			}
			matchesChan <- match

		}(matchCount)
		if !*flagWins {
			done = done || (matchCount+1 >= numMatchesToPlay)
		}
		klog.V(1).Infof("Started match %d, done=%v (wins so far=%d)", matchCount, done, wins)
	}

	// Wait matches to complete.
	wg.Wait()
	spinner.Done()
	fmt.Println("done.")
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

func renameToFinal(filename string) error {
	if _, err := os.Stat(filename); err == nil {
		err = os.Rename(filename, backupName(filename))
		if err != nil {
			return errors.Wrapf(err, "failed backing up, while renaming %q to %q", filename, backupName(filename))
		}
	}
	err := os.Rename(temporaryName(filename), filename)
	if err != nil {
		return errors.Wrapf(err, "failed renaming generate file to final name, while renaming %q to %q", temporaryName(filename), filename)
	}
	return nil
}

// Load matches, and automatically build boards.
func loadMatches(ctx context.Context, results chan<- *Match) {
	// Makes sure we close the channel at exit, along with a report of loaded matches.
	var matchesCount, matchesIdx, numWins int
	defer func() {
		klog.Infof("%d matches loaded (%d wins), %d used (not filtered out)", matchesIdx, numWins, matchesCount)
		close(results)
	}()

	filenames, err := filepath.Glob(*flagLoadMatches)
	if err != nil {
		klog.Errorf("Invalid pattern %q for loading matches: %v", *flagLoadMatches, err)
		return
	}
	if len(filenames) == 0 {
		klog.Errorf("Did not find any files matching %q", *flagLoadMatches)
		return
	}

LoopFilenames:
	for _, filename := range filenames {
		klog.V(1).Infof("Scanning file %s\n", filename)
		file, err := os.Open(filename)
		if err != nil {
			klog.Errorf("Cannot open %q for reading: %v", filename, err)
			return
		}
		dec := gob.NewDecoder(file)
		for *flagNumMatches == 0 || matchesCount < *flagNumMatches {
			if ctx.Err() != nil {
				klog.Infof("Processing interrupted (context cancelled): %v", ctx.Err())
				return
			}
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
}

// Report on matches as they are being played/generated.
func reportMatches(results <-chan *Match) (matches []*Match) {
	// Read results.
	totalWins := [3]int{0, 0, 0}
	totalMoves := 0
	ui := cli.New(true, false)
	for match := range results {
		// Accounting.
		board := match.FinalBoard()
		if board == nil {
			continue
		}
		matches = append(matches, match)
		wins := board.Derived.Wins
		swapped := match.PlayersIdx[0] == 1
		if swapped {
			// Players were swapped.
			wins[0], wins[1] = wins[1], wins[0]
		}
		if *flagPrint {
			if swapped {
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
	if count == 0 {
		return
	}
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
