package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"golang.org/x/sync/errgroup"
	"k8s.io/klog/v2"
	"runtime"
	"sync"
	"time"
)

var (
	flagNumMatches = flag.Int("num_matches", 100, "Number of matches to play per iteration, "+
		"between training the model. If the new model is not better than the old, repeat again. ")
	flagMaxMoves = flag.Int(
		"max_moves", DefaultMaxMoves, "Max moves before game is assumed to be a draw.")
	flagParallelism = flag.Int("parallelism", 0, "If > 0 ignore GOMAXPROCS and play "+
		"these many matches simultaneously.")
	flagPrintSteps = flag.Bool("print_steps", false, "Print board at each step. "+
		"Very verbose, and you probably want to set flagParallelism to 1.")
)

var (
	stepUI   = cli.New(true, false)
	muStepUI sync.Mutex
)

type CollectExamples struct {
	examples []Example
	mu       sync.Mutex
}

// runMatches and collect training examples.
func runMatches(ctx context.Context) ([]Example, error) {
	numMatches := *flagNumMatches
	var collect CollectExamples
	var wg errgroup.Group
	parallelism := getParallelism()
	wg.SetLimit(parallelism)
	var count int
	start := time.Now()
	printUpdate := func() {
		elapsed := time.Since(start)
		fmt.Printf("\r\tRunning matches (parallelism=%d): %5d of %d finished in %s\x1b[0K", parallelism, count, numMatches, elapsed)
	}
	printUpdate()
	for matchIdx := range numMatches {
		wg.Go(func() error {
			err := runMatch(ctx, matchIdx, &collect)
			if err == nil && ctx.Err() == nil {
				printUpdate()
				count++
			}
			return err
		})
	}
	err := wg.Wait()
	if err != nil {
		fmt.Printf("\n")
		return nil, err
	}
	if ctx.Err() != nil {
		fmt.Printf("\nInterrupted: %s\n", ctx.Err())
		return nil, nil
	}
	printUpdate()
	fmt.Println()
	return collect.examples, nil
}

// runMatch from start to end, and send the resulting moves and policies through the given channel.
// It returns nil is the ctx was cancelled at any point.
func runMatch(ctx context.Context, matchNum int, output *CollectExamples) error {
	if ctx.Err() != nil {
		// Trainer already interrupted.
		return nil
	}
	if klog.V(1).Enabled() {
		klog.Infof("Starting match %d", matchNum)
		defer klog.Infof("Finished match %d", matchNum)
	}
	board := NewBoard()
	board.MaxMoves = *flagMaxMoves
	examples := make([]Example, 0, board.MaxMoves)
	playerNums := make([]PlayerNum, 0, board.MaxMoves)
	matchName := fmt.Sprintf("Match-%05d", matchNum)

	// Run match.
	for !board.IsFinished() {
		if ctx.Err() != nil {
			klog.Infof("Match %d interrupted: %s", matchNum, ctx.Err())
			return nil
		}
		playerNum := board.NextPlayer
		if klog.V(2).Enabled() {
			klog.Infof(
				"\n\n%s: %s at turn %d (#valid actions=%d)\n\n",
				matchName, playerNum, board.MoveNumber, len(board.Derived.Actions))
		}
		action, nextBoard, _, actionLabels := aiPlayer.Play(board)
		if bootstrapAiPlayer != nil && playerNum == PlayerNum(matchNum%2) {
			// Actually, play using the bootstrapAI, but keep the actionLabels from the AlphaZero AI.
			action, nextBoard, _, _ = bootstrapAiPlayer.Play(board)
		}
		if !action.IsSkipAction() {
			// Record example only of non-skip actions.
			examples = append(examples, Example{
				board:        board,
				valueLabel:   0, // We will use as label the final result of the match, so to be set in the end.
				policyLabels: actionLabels,
			})
			playerNums = append(playerNums, playerNum)
		}
		board.ClearNextBoardsCache()
		board = nextBoard
		if *flagPrintSteps {
			muStepUI.Lock()
			fmt.Printf("%s, move #%d:\n", matchName, board.MoveNumber)
			fmt.Printf("\taction:\t%s\n", action)
			fmt.Printf("\tpolicy:\t%v\n", actionLabels)
			fmt.Println()
			stepUI.PrintBoard(board)
			fmt.Println()
			fmt.Println("------------------")
			muStepUI.Unlock()
		}
	}

	// Re-score examples to whoever won, if it was not a draw.
	// (Scores are already set to 0, so if it was a draw there is nothing to do).
	if _, endScore := ai.IsEndGameAndScore(board); endScore != 0 {
		var scoresPerPlayers [2]float32
		scoresPerPlayers[board.NextPlayer] = endScore
		scoresPerPlayers[1-board.NextPlayer] = -endScore
		for ii := range examples {
			examples[ii].valueLabel = scoresPerPlayers[playerNums[ii]]
		}
	}

	// Copy over final examples:
	output.mu.Lock()
	defer output.mu.Unlock()
	output.examples = append(output.examples, examples...)
	return nil
}

// getParallelism returns the parallelism.
func getParallelism() (parallelism int) {
	parallelism = runtime.GOMAXPROCS(0)
	if *flagParallelism > 0 {
		parallelism = *flagParallelism
	}
	return
}
