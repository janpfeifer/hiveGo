package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/janpfeifer/hiveGo/internal/ai/gomlx"
	"github.com/janpfeifer/hiveGo/internal/ai/gomlx/bench"
	"github.com/janpfeifer/hiveGo/internal/players"
	_ "github.com/janpfeifer/hiveGo/internal/players/default"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/cli"
	"github.com/janpfeifer/hiveGo/internal/ui/spinning"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"math/rand/v2"
	"strings"
	"time"
)

var (
	_ = fmt.Printf

	flagHotseat   = flag.Bool("hotseat", false, "Hotseat match: human vs human")
	flagWatch     = flag.Bool("watch", false, "Watch mode: AI vs AI playing")
	flagFirst     = flag.String("first", "", "Who plays first: human or ai. Default is random.")
	flagAIConfig  = flag.String("config", "a0fnn=#0,mcts,max_time=3s,temperature=0.2", "AI configuration against which to play")
	flagAIConfig2 = flag.String("config2", "a0fnn=#0,mcts,max_time=3s,temperature=0.2", "Second AI configuration, if playing AI vs AI with --watch")
	flagMaxMoves  = flag.Int(
		"max_moves", DefaultMaxMoves, "Max moves before game is considered a draw.")
	flagQuiet        = flag.Bool("quiet", false, "Quiet mode for when watching AI play, only the actions and the last board position is printed.")
	flagSaveFeatures = flag.String("save_features", "", "Path to save extracted benchmark dataset features from the match.")

	// aiPlayers: if nil, it's a human playing.
	aiPlayers = [2]players.Player{nil, nil}
	matchId   = uint64(0)
	matchName = "The Match"

	globalCtx = context.Background()
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	if *flagMaxMoves <= 0 {
		klog.Fatalf("Invalid --max_moves=%d", *flagMaxMoves)
	}

	// Capture Control+C
	var cancel func()
	globalCtx, cancel = context.WithCancel(context.Background())
	spinning.SafeInterrupt(cancel, 3*time.Second)
	defer cancel()

	// Create players.
	createPlayers()

	// Create board and UI.
	board := NewBoard()
	board.MaxMoves = *flagMaxMoves
	ui := cli.New(true, false)

	var recordedDataset *bench.MatchDataset
	var fnnModel *gomlx.FNN
	var a0Model *gomlx.AlphaZeroFNN

	if *flagSaveFeatures != "" {
		recordedDataset = &bench.MatchDataset{}
		fnnModel = gomlx.NewFNN()
		a0Model = gomlx.NewAlphaZeroFNN()
	}

	// Loop over match.
	for !board.IsFinished() && globalCtx.Err() == nil {
		if newBoard, skip := ui.CheckNoAvailableAction(board); skip {
			board = newBoard
			continue
		}

		if recordedDataset != nil {
			step := bench.StepFeatures{}
			nextBoards := board.TakeAllActions()
			if len(nextBoards) > 0 {
				// Capture FNN inputs for all next legal boards (batched like AlphaBeta search)
				fnnTensors := fnnModel.CreateInputs(nextBoards)
				for _, t := range fnnTensors {
					step.FNNInputs = append(step.FNNInputs, bench.FromTensor(t))
				}
				// Capture A0FNN value inputs for all next legal boards
				a0ValTensors := a0Model.CreateValueInputs(nextBoards[0])
				if len(a0ValTensors) == 2 {
					// Update batch size for all next boards
					a0ValTensors = []*tensors.Tensor{
						a0Model.CreateBoardsFeatures(nextBoards, 0),
						tensors.FromScalar(int32(len(nextBoards))),
					}
				}
				for _, t := range a0ValTensors {
					step.A0ValInputs = append(step.A0ValInputs, bench.FromTensor(t))
				}
			}
			// Capture A0FNN policy inputs
			a0PolTensors := a0Model.CreatePolicyInputs([]*Board{board})
			for _, t := range a0PolTensors {
				step.A0PolInputs = append(step.A0PolInputs, bench.FromTensor(t))
			}
			recordedDataset.Steps = append(recordedDataset.Steps, step)
		}

		aiPlayer := aiPlayers[board.NextPlayer]
		if aiPlayer == nil {
			newBoard, err := ui.RunNextMove(board)
			if err != nil {
				klog.Exitf("Failed to run match: %+v", err)
			}
			// Release data used during search of AI plays.
			board.ClearNextBoardsCache()
			board = newBoard

		} else {
			// AI plays.
			if *flagWatch && !*flagQuiet {
				ui.Print(board, false)
				fmt.Printf("\t%s action: ", aiPlayer)
			} else {
				fmt.Printf("AI: %s\n", aiPlayer)
				ui.PrintSpacedPlayer(board)
			}

			s := spinning.New(globalCtx)
			action, newBoard, score, _ := aiPlayer.Play(board)
			s.Done()
			fmt.Printf(" %s (score=%.3f)\n", action, score)
			// Release data used during search of AI plays.
			board.ClearNextBoardsCache()
			board = newBoard
			fmt.Println()
		}
	}
	if globalCtx.Err() != nil {
		fmt.Printf("\nMatch interrupted: %s\n", globalCtx.Err())
		return
	}

	if recordedDataset != nil && len(recordedDataset.Steps) > 0 {
		err := recordedDataset.Save(*flagSaveFeatures)
		if err != nil {
			klog.Errorf("Failed to save recorded dataset to %q: %+v", *flagSaveFeatures, err)
		} else {
			fmt.Printf("Successfully saved %d step features to %q\n", len(recordedDataset.Steps), *flagSaveFeatures)
		}
	}

	fmt.Printf("> %s\n", lipgloss.NewStyle().Bold(true).Render(board.FinishReason()))
	ui.Print(board, false)
	ui.PrintWinner(board)
	winner := board.Winner()
	if winner != PlayerInvalid && aiPlayers[winner] != nil {
		fmt.Printf("Winner AI: %s\n", aiPlayers[winner])
	}
}

// createPlayers in aiPlayers.
func createPlayers() {
	if *flagHotseat && *flagWatch {
		klog.Fatalf("--hotseat and --watch cannot be used together")
	}
	if *flagHotseat {
		// Both players are human, nothing to do.
		return
	}

	// Create AI player:
	var aiPlayerNum PlayerNum
	if strings.ToLower(*flagFirst) == "human" {
		aiPlayerNum = 1
	} else if strings.ToLower(*flagFirst) == "ai" {
		aiPlayerNum = 0
	} else if *flagFirst == "" {
		// Random:
		aiPlayerNum = PlayerNum(rand.IntN(2))
	} else {
		exceptions.Panicf("invalid --first=%q, only valid values are \"human\" or \"ai\"", *flagFirst)
	}
	aiPlayers[aiPlayerNum] = must.M1(players.New(*flagAIConfig))
	if !*flagWatch {
		return
	}

	// Create second AI
	otherPlayerNum := 1 - aiPlayerNum
	aiPlayers[otherPlayerNum] = must.M1(players.New(*flagAIConfig2))
	return
}
