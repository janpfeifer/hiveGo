package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/players"
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
	flagNumCompareMatches = flag.Int("num_compare", 40, "Number of matches to play to check whether a model is better than another.")
	flagMaxMoves          = flag.Int(
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
func runMatches(ctx context.Context, numMatches int, aPlayer, bPlayer *players.SearcherScorer) (aWins, bWins, draws int, examples []Example, err error) {
	var collect CollectExamples
	var wg errgroup.Group
	parallelism := getParallelism()
	wg.SetLimit(parallelism)
	var count int
	start := time.Now()
	printUpdate := func() {
		elapsed := time.Since(start)
		fmt.Printf("\r\tRunning matches (parallelism=%d): %5d of %d finished (%d/%d/%d A-Wins/B-Wins/Draws) in %s\x1b[0K",
			parallelism, count, numMatches, aWins, bWins, draws, elapsed)
	}
	printUpdate()

	for matchIdx := range numMatches {
		wg.Go(func() error {
			var matchPlayers [2]*players.SearcherScorer
			isSwapped := matchIdx%2 == 1
			if !isSwapped {
				matchPlayers[0] = aPlayer
				matchPlayers[1] = bPlayer
			} else {
				matchPlayers[1] = aPlayer
				matchPlayers[0] = bPlayer
			}
			winner, err := runMatch(ctx, matchIdx, matchPlayers, &collect)
			if err == nil && ctx.Err() == nil {
				// Record winner.
				if winner == PlayerInvalid {
					draws++
				} else {
					if isSwapped {
						winner = 1 - winner
					}
					if winner == PlayerFirst {
						aWins++
					} else {
						bWins++
					}
				}
				count++
				printUpdate()
			}
			return err
		})
	}
	err = wg.Wait()
	printUpdate()
	fmt.Println()
	if err != nil {
		return
	}
	if ctx.Err() != nil {
		fmt.Printf("Interrupted: %s\n", ctx.Err())
		return
	}
	fmt.Printf("\t- %d matches, %d draws (%.2f%%), %d new training examples.\n",
		numMatches, draws, 100*float32(draws)/float32(numMatches), len(collect.examples))
	examples = collect.examples
	return
}

// runMatch from start to end, and send the resulting moves and policies through the given channel.
// It returns nil is the ctx was cancelled at any point.
func runMatch(ctx context.Context, matchNum int, players [2]*players.SearcherScorer, output *CollectExamples) (winner PlayerNum, err error) {
	if ctx.Err() != nil {
		// Trainer already interrupted.
		return PlayerInvalid, nil
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
			return PlayerInvalid, nil
		}
		playerNum := board.NextPlayer
		player := players[playerNum]
		if klog.V(2).Enabled() {
			klog.Infof(
				"\n\n%s: %s at turn %d (#valid actions=%d)\n\n",
				matchName, playerNum, board.MoveNumber, len(board.Derived.Actions))
		}
		action, nextBoard, _, actionLabels := player.Play(board)
		if player == bootstrapAiPlayer {
			// If played using bootstrapAI, collect actionsLabels from AlphaZero player instead.
			_, _, _, actionLabels = aiPlayer.Play(board)
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
		if endScore > 0 {
			winner = board.NextPlayer
		} else {
			winner = 1 - board.NextPlayer
		}
		var scoresPerPlayers [2]float32
		scoresPerPlayers[board.NextPlayer] = endScore
		scoresPerPlayers[1-board.NextPlayer] = -endScore
		for ii := range examples {
			examples[ii].valueLabel = scoresPerPlayers[playerNums[ii]]
			//fmt.Printf("\texample #%d: playerNum=%d, score=%.1f\n", ii, playerNums[ii], examples[ii].valueLabel)
		}
	} else {
		winner = PlayerInvalid
	}

	// Copy over final examples:
	output.mu.Lock()
	defer output.mu.Unlock()
	output.examples = append(output.examples, examples...)
	return winner, nil
}

// comparePlayers pitch two players and compare which one has more wins.

// getParallelism returns the parallelism.
func getParallelism() (parallelism int) {
	parallelism = runtime.GOMAXPROCS(0)
	if *flagParallelism > 0 {
		parallelism = *flagParallelism
	}
	return
}
