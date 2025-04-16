package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/players"
	_ "github.com/janpfeifer/hiveGo/internal/players/default"
	"github.com/janpfeifer/hiveGo/internal/profilers"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"github.com/janpfeifer/hiveGo/internal/ui/spinning"
	"github.com/janpfeifer/must"
	"golang.org/x/sync/errgroup"
	"k8s.io/klog/v2"
	"runtime"
	"strings"
	"sync"
	"time"
)

var (
	flagPlayer1Config = flag.String("ai1", "", "1st player configuration.")
	flagPlayer2Config = flag.String("ai2", "", "2nd player configuration.")
	flagNumMatches    = flag.Int("num_matches", 100, "Number of matches to play.")
	flagParallelism   = flag.Int("parallelism", 0, "If > 0 ignore GOMAXPROCS and play "+
		"these many matches simultaneously.")
	flagPrintSteps = flag.Bool("print_steps", false, "Print board at each step. "+
		"Very verbose, and you probably want to set flagParallelism to 1.")
	flagMaxMoves = flag.Int(
		"max_moves", state.DefaultMaxMoves, "Max moves before game is assumed to be a draw.")
)

// Globals
var (
	// globalCtx used everywhere. It is cancelled when the program is about to exit either by
	// an interrupt (ctrl+C) or by reaching the end.
	globalCtx = context.Background()
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagPlayer1Config == "" || *flagPlayer2Config == "" {
		klog.Fatal("You must configure both players to compare with flags -ai1 and -ai2")
	}

	// Capture Control+C
	var globalCancel func()
	globalCtx, globalCancel = context.WithCancel(context.Background())
	spinning.SafeInterrupt(globalCancel, 5*time.Second)
	defer globalCancel()

	// Profilers: HTTP profiler server and CPU profile.
	profilers.Setup(globalCtx)
	defer profilers.OnQuit()

	aiPlayers := must.M1(createAIPlayers())
	defer func() {
		aiPlayers[0], aiPlayers[1] = nil, nil
	}()
	must.M(runMatches(globalCtx, aiPlayers))
}

func createAIPlayers() (aiPlayers [2]players.Player, err error) {
	for playerIdx, config := range [2]string{*flagPlayer1Config, *flagPlayer2Config} {
		klog.V(1).Infof("Creating AI for player #%d from %q", playerIdx, config)
		aiPlayers[playerIdx], err = players.New(config)
		if err != nil {
			return
		}
	}
	return
}

type Results struct {
	mu                   sync.Mutex
	start                time.Time
	winsAs1st, winsAs2nd [2]int
	draws                [2]int
	played, total        int
}

func (r *Results) String() string {
	var parts []string
	parts = append(parts, fmt.Sprintf("Played %d of %d: ", r.played, r.total))
	for playerIdx := range 2 {
		parts = append(parts,
			fmt.Sprintf("AI-%d: %d Wins (1st: %d, 2nd: %d) / ",
				playerIdx+1, r.winsAs1st[playerIdx]+r.winsAs2nd[playerIdx],
				r.winsAs1st[playerIdx], r.winsAs2nd[playerIdx]))
	}
	parts = append(parts, fmt.Sprintf("%d draws (%d AI-1 as 1st, %d AI-2 as 1st) - ",
		r.draws[0]+r.draws[1], r.draws[0], r.draws[1]))
	parts = append(parts, fmt.Sprintf("%s", time.Since(r.start)))
	parts = append(parts, "[0K")
	return strings.Join(parts, "")
}

func runMatches(ctx context.Context, aiPlayers [2]players.Player) error {
	r := &Results{
		start: time.Now(),
		total: *flagNumMatches,
	}
	var wg errgroup.Group
	parallelism := getParallelism()
	wg.SetLimit(parallelism)
	fmt.Printf("\r%s", r)

	for matchIdx := range r.total {
		wg.Go(func() error {
			matchPlayers := aiPlayers
			isSwapped := matchIdx%2 == 1
			player1st := 0
			if isSwapped {
				matchPlayers[0], matchPlayers[1] = matchPlayers[1], matchPlayers[0]
				player1st = 1
			}
			winner, err := runMatch(ctx, matchIdx, matchPlayers)
			if err != nil || ctx.Err() != nil {
				return err
			}
			// Record winner.
			r.mu.Lock()
			defer r.mu.Unlock()
			if winner == state.PlayerInvalid {
				r.draws[player1st]++
			} else {
				if isSwapped {
					winner = 1 - winner
				}
				if int(winner) == player1st {
					r.winsAs1st[winner]++
				} else {
					r.winsAs2nd[winner]++
				}
			}
			r.played++
			fmt.Printf("\r%s", r)
			return nil
		})
	}
	err := wg.Wait()
	fmt.Printf("\r%s", r)
	fmt.Println()
	if ctx.Err() != nil {
		fmt.Printf("Interrupted: %s\n", ctx.Err())
		return nil
	}
	return err
}

var (
	stepUI   = cli.New(true, false)
	muStepUI sync.Mutex
)

func runMatch(ctx context.Context, matchNum int, aiPlayers [2]players.Player) (winner state.PlayerNum, err error) {
	if ctx.Err() != nil {
		// Trainer already interrupted.
		return state.PlayerInvalid, nil
	}
	if klog.V(1).Enabled() {
		klog.Infof("Starting match %d", matchNum)
		defer klog.Infof("Finished match %d", matchNum)
	}
	board := state.NewBoard()
	board.MaxMoves = *flagMaxMoves
	matchName := fmt.Sprintf("Match-%05d", matchNum)

	// Run match.
	for !board.IsFinished() {
		if ctx.Err() != nil {
			klog.V(1).Infof("Match %d interrupted: %s", matchNum, ctx.Err())
			return state.PlayerInvalid, nil
		}
		playerNum := board.NextPlayer
		player := aiPlayers[playerNum]
		if klog.V(2).Enabled() {
			klog.Infof(
				"\n\n%s: %s at turn %d (#valid actions=%d)\n\n",
				matchName, playerNum, board.MoveNumber, len(board.Derived.Actions))
		}
		action, nextBoard, _, actionLabels := player.Play(board)
		board.ClearNextBoardsCache()
		if *flagPrintSteps {
			muStepUI.Lock()
			fmt.Printf("%s, move #%d\n", matchName, board.MoveNumber)
			fmt.Println()
			stepUI.PrettyPrintActionsWithPolicy(board, actionLabels, action, 5)
			fmt.Println()
			stepUI.PrintBoard(nextBoard)
			fmt.Println()
			fmt.Println("------------------")
			muStepUI.Unlock()
		}
		board = nextBoard
	}

	// Re-score examples to whoever won, if it was not a draw.
	// (Scores are already set to 0, so if it was a draw there is nothing to do).
	if _, endScore := ai.IsEndGameAndScore(board); endScore != 0 {
		if endScore > 0 {
			winner = board.NextPlayer
		} else {
			winner = 1 - board.NextPlayer
		}
	} else {
		winner = state.PlayerInvalid
	}
	return winner, nil
}

// getParallelism returns the parallelism.
func getParallelism() (parallelism int) {
	parallelism = runtime.GOMAXPROCS(0)
	if *flagParallelism > 0 {
		parallelism = *flagParallelism
	}
	return
}
