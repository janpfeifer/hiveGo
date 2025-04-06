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
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
	"slices"
	"strings"
	"time"
	// "github.com/janpfeifer/hiveGo/ai/tfddqn"
	//ab "github.com/janpfeifer/hiveGo/internal/searchers/alphabeta"
	//"github.com/janpfeifer/hiveGo/internal/searchers/mcts"
)

var _ = fmt.Printf

var (
	flagCpuProfile = flag.String("cpu_profile", "", "write cpu profile to `file`")

	flagPlayers = flag.String("ai", "linear,ab,max_depth=2;same", "Configuration string for AI players. "+
		"Each AI player configuration is separated by \";\", and options within an AI config are separated "+
		"by \",\". The special value \"same\" repeats the previous AI configuration.")

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

	flagTrain = flag.String("train", "same", "Set to the AI model definition to be trained. "+
		"If not empty, train the given model. "+
		"If the value is \"same\", it uses teh first AI model configured with -ai.")
	flagTrainLoops      = flag.Int("train_loops", 1, "After acquiring data for all matches, how many times to loop the training over it.")
	flagTrainValidation = flag.Int("train_validation", 0, "Percentage (int value) of matches used for validation (evaluation only).")
	flagRescore         = flag.Bool("rescore", false, "If to rescore matches.")
	flagDistill         = flag.Bool("distill", false,
		"If set it will simply distill from the first model set by -ai to the second, without searching for best moves.")
	flagTrainWithEndScore = flag.Float64("train_with_end_score",
		0, "If > 0, it will use the final match score across all moves. "+
			"Also known as Monte Carlo learning. Values between 0 and 1 are interpreted as weight "+
			"of the final match score to use as label for each board position of the match.")
	flagTrainBoardsBufferSize = flag.Int("train_buffer_size",
		500, "Size of board positions to keep in a rotating buffer during training.")
	flagTrainStepsPerMatch = flag.Int("train_steps_per_match",
		0, "Number of steps of call to Learn() per match read/played. If set to 0 it will "+
			"divide the --train_buffer_size by the learner batch size. Batches are created with random "+
			"sampling with replacement strategy.")

	flagParallelism = flag.Int("parallelism", 0, "If > 0 ignore GOMAXPROCS and play "+
		"these many matches simultaneously.")
	flagMaxAutoBatch = flag.Int("max_auto_batch", 0, "If > 0 ignore at most do given value of "+
		"auto-batch for evaluations.")

	flagContinuosRescoreAndTrain = flag.Bool("rescore_and_train", false, "If set, continuously rescore and train matches.")
	flagRescoreAndTrainPoolSize  = flag.Int("rescore_and_train_pool_size", 10000,
		"How many board positions to keep in pool (in a rotating buffer) used to train. ")
	flagRescoreAndTrainIssueLearn = flag.Int("rescore_and_train_issue_learn", 10,
		"After how many rescored matches/action to issue another learning mini-batch.")

	flagProfiler = flag.Int("prof", -1, "If set, runs the profile at the given port.")

	// AI for the players. If their configuration is exactly the same, they will point to the same object.
	// trainingAI is the AI set to be trained (if one is set) -- it is nil, if none was set.
	aiPlayers  []*players.SearcherScorer
	trainingAI *players.SearcherScorer

	// globalCtx used everywhere. It is cancelled when the program is about to exit either by
	// an interrupt (ctrl+C) or by reaching the end.
	globalCtx = context.Background()
)

// releaseGlobals should free all player and training information.
// Used at the end of the program, to help profile for memory leaks.
func releaseGlobals() {
	aiPlayers = nil
	trainingAI = nil
}

// main orchestrates playing, loading, rescoring, saving and training of matches.
func main() {
	klog.InitFlags(nil)
	flag.Parse()

	// Capture Control+C
	var globalCancel func()
	globalCtx, globalCancel = context.WithCancel(context.Background())
	spinning.SafeInterrupt(globalCancel, 5*time.Second)
	defer globalCancel()

	// Optionally run profiler, and keep program alive on exit.
	if *flagProfiler >= 0 {
		addr := fmt.Sprintf("localhost:%d", *flagProfiler)
		fmt.Printf("Starting profiler on %s/debug/pprof\n", addr)
		fmt.Printf("- You can access it with: $ go tool pprof %s/debug/pprof/heap\n", addr)
		fmt.Printf("- Program will be kept alive on end, you will have to interrupt it (Ctrl+C) to exit\n")
		go func() {
			klog.Fatal(http.ListenAndServe(addr, nil))
		}()
		defer func() {
			// Release/free everything, there should be very little allocations left.
			releaseGlobals()
			for _ = range 10 {
				runtime.GC()
			}
			fmt.Printf("- Program finished: kept alive with profiler opened at %s/debug/pprof\n", addr)
			fmt.Printf("- Interrupt (Ctrl+C) to exit\n")
			<-globalCtx.Done()
			fmt.Printf("... exiting ...\n")
		}()
	}

	if *flagCpuProfile != "" {
		whenDone := createCPUProfile()
		defer whenDone()
	}

	if *flagMaxMoves <= 0 {
		klog.Fatalf("Invalid --max_moves=%d", *flagMaxMoves)
	}

	// Create AI players, and training AI if one is configured.
	createAIPlayers()

	if *flagContinuosPlayAndTrain {
		if trainingAI == nil {
			klog.Fatal("For continuous training with -play_and_train you need to set an AI model to train with -train=...")
		}
		// (a) Continuous play and train uses a separate flow:
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
	if trainingAI != nil {
		trainFromMatches(matches)
		return
	}
	if *flagContinuosRescoreAndTrain {
		err := rescoreAndTrain(globalCtx, matches)
		if err != nil {
			globalCancel()
			klog.Fatalf("Failed to continuously rescore and train: %+v", err)
		}
		return
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
	configs := strings.Split(*flagPlayers, ";")
	configs = slices.DeleteFunc(configs, func(s string) bool { return s == "" })
	aiPlayers = make([]*players.SearcherScorer, len(configs))

	for idxConfig, config := range configs {
		klog.V(1).Infof("Creating AI #%d from %q", idxConfig, config)
		if config == "same" {
			if idxConfig == 0 {
				klog.Fatalf("First AI player cannot be configured as %q", config)
			}
			aiPlayers[idxConfig] = aiPlayers[idxConfig-1]
			continue
		}
		var err error
		aiPlayers[idxConfig], err = players.New(config)
		if err != nil {
			klog.Fatalf("Failed to create AI #%d from %q: %+v", idxConfig, config, err)
		}
	}

	config := *flagTrain
	if config != "" {
		if config == "same" {
			if len(aiPlayers) == 0 {
				klog.Fatalf("No AI player configured, cannot use %q for training AI", config)
			}
			trainingAI = aiPlayers[0]
		} else {
			var err error
			trainingAI, err = players.New(config)
			if err != nil {
				klog.Fatalf("Failed to create training AI from %q: %+v", config, err)
			}
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
