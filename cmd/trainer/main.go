package main

import (
	"context"
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/players"
	_ "github.com/janpfeifer/hiveGo/internal/players/default"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/janpfeifer/hiveGo/internal/ui/spinning"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"os"
	"runtime"
	"runtime/pprof"
	// "github.com/janpfeifer/hiveGo/ai/tfddqn"
	//ab "github.com/janpfeifer/hiveGo/internal/searchers/alphabeta"
	//"github.com/janpfeifer/hiveGo/internal/searchers/mcts"
)

var _ = fmt.Printf

var (
	flagCpuProfile = flag.String("cpu_profile", "", "write cpu profile to `file`")

	flagPlayers = [2]*string{
		flag.String("ai0", "linear,ab", "Configuration string for ai playing as the starting player."),
		flag.String("ai1", "linear,ab", "Configuration string for ai playing as the second player."),
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
	flagDiscount        = flag.Float64("discount", 0.99, "Discount multiplier when using \"future\" V_{target}(t) = discount*V(t+1) scores to current one. This should be ~ \\lambda^max_depth used by the AI.")
	flagDistill         = flag.Bool("directRescoreMatch", false,
		"If set it will simply directRescoreMatch from --ai1 to --ai0, without searching for best moves.")
	flagLearnWithEndScore = flag.Bool("learn_with_end_score",
		true, "If true will use the final score to learn.")
	flagTrainingBoardsBufferSize = flag.Int("train_buffer_size",
		500, "Size of board positions to keep in a rotating buffer during training.")

	flagParallelism = flag.Int("parallelism", 0, "If > 0 ignore GOMAXPROCS and play "+
		"these many matches simultaneously.")
	flagMaxAutoBatch = flag.Int("max_auto_batch", 0, "If > 0 ignore at most do given value of "+
		"auto-batch for evaluations.")

	flagContinuosRescoreAndTrain = flag.Bool("rescore_and_train", false, "If set, continuously rescore and train matches.")
	flagRescoreAndTrainPoolSize  = flag.Int("rescore_and_train_pool_size", 10000,
		"How many board positions to keep in pool (in a rotating buffer) used to train. ")
	flagRescoreAndTrainIssueLearn = flag.Int("rescore_and_train_issue_learn", 10,
		"After how many rescored matches/action to issue another learning mini-batch.")

	// AI for the players. If their configuration is exactly the same, they will point to the same object.
	aiPlayers [2]*players.SearcherScorer

	// globalCtx used everywhere. It is cancelled when the program is about to exit either by
	// an interrupt (ctrl+C) or by reaching the end.
	globalCtx = context.Background()
)

// main orchestrates playing, loading, rescoring, saving and training of matches.
func main() {
	klog.InitFlags(nil)
	flag.Parse()

	// Capture Control+C
	var globalCancel func()
	globalCtx, globalCancel = context.WithCancel(context.Background())
	spinning.SafeInterrupt(globalCancel)
	defer globalCancel()

	if *flagCpuProfile != "" {
		whenDone := createCPUProfile()
		defer whenDone()
	}

	if *flagMaxMoves <= 0 {
		klog.Fatalf("Invalid --max_moves=%d", *flagMaxMoves)
	}
	createAIPlayers()

	// Continuous play and train uses a separate flow:
	if *flagContinuosPlayAndTrain {
		err := playAndTrain(globalCtx)
		if err != nil {
			globalCancel()
			klog.Errorf("Failed to continuous play and train: %+v", err)
		}
		return
	}

	// Run/load matches: the new or loaded matches will be fed into matchesChan.
	// To interrupt at any time, cancel globalCtx (with globalCancel).
	matchesChan := make(chan *Match)
	if *flagLoadMatches != "" {
		go loadMatches(globalCtx, matchesChan)
	} else {
		go runMatches(globalCtx, matchesChan)
	}
	if *flagRescore {
		rescored := make(chan *Match)
		go rescoreMatches(matchesChan, rescored)
		matchesChan = rescored
	}

	// Collect resulting matches, optionally reporting on them.
	var matches []*Match
	if !*flagRescore {
		matches = reportMatches(matchesChan)
	} else {
		for match := range matchesChan {
			matches = append(matches, match)
		}
	}

	if *flagSaveMatches != "" {
		err := saveMatches(matches)
		if err != nil {
			klog.Fatalf("Failed to save matches in %s: %+v", *flagSaveMatches, err)
		}
	}
	if *flagTrain {
		trainFromMatches(matches)
	}
	if *flagContinuosRescoreAndTrain {
		rescoreAndTrain(matches)
	}
}

// getParallelism returns the parallelism.
func getParallelism() (parallelism int) {
	parallelism = runtime.GOMAXPROCS(0)
	if *flagParallelism > 0 {
		parallelism = *flagParallelism
	}
	return
}

func createAIPlayers() {
	// Create AI players. If they are the same, reuse -- sharing same TF Session can be more
	// efficient.
	klog.V(1).Infof("Creating player 0 from %q", *flagPlayers[0])
	var err error
	aiPlayers[0], err = players.New(*flagPlayers[0])
	if err != nil {
		klog.Fatalf("Failed to create player 0: %+v", err)
	}
	if *flagPlayers[1] == *flagPlayers[0] {
		aiPlayers[1] = aiPlayers[0]
		isSamePlayer = true
		klog.V(1).Infof("Player 1 is the same as player 0, reusing AI player object.")
	} else {
		klog.V(1).Infof("Creating player 1 from %q", *flagPlayers[1])
		aiPlayers[1], err = players.New(*flagPlayers[1])
		if err != nil {
			klog.Fatalf("Failed to create player 1: %+v", err)
		}
	}
}

func createCPUProfile() func() {
	f, err := os.Create(*flagCpuProfile)
	if err != nil {
		klog.Fatal("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		klog.Fatal("could not start CPU profile: ", err)
	}
	return pprof.StopCPUProfile
}

// Save all given matches to location given --save_matches.
// It creates file into a temporary file first, and move
// to final destination once finished.
func saveMatches(matches []*Match) error {
	file := openWriterAndBackup(*flagSaveMatches)
	enc := gob.NewEncoder(file)
	for _, match := range matches {
		err := match.Encode(enc)
		if err != nil {
			return errors.WithMessagef(err, "while encoding match %d", match.MatchNum)
		}
	}
	err := file.Close()
	if err != nil {
		return errors.Wrapf(err, "failed to close saved matches in %q", *flagSaveMatches)
	}
	err = renameToFinal(*flagSaveMatches)
	if err != nil {
		return errors.WithMessagef(err, "while saving matches to %q", *flagSaveMatches)
	}
	return nil
}
