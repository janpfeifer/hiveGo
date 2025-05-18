// a0-trainer is a command line tool to orchestrate the training of Alpha-Zero based models.
//
// It works by:
//  1. Self-play a bunch of matches and collect training data.
//  2. Train a copy of the model with the new data.
//  3. Pitch the previous model with the newly trained one.
//  4. If newly trained model is not > 10% better than previous model, discard the newly trained model
//     (but not the data accumulated so far) and return to (1) and to collect more data.
//  5. Else newly trained model becomes the new current model (and discard the previous one),
//     checkpoint (save) it, and restart at 1.
//
// See -help for flags.
package main

import (
	"context"
	"flag"
	"fmt"
	"time"

	_ "github.com/janpfeifer/hiveGo/internal/players/default"
	"github.com/janpfeifer/hiveGo/internal/profilers"
	"github.com/janpfeifer/hiveGo/internal/ui/spinning"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
)

// Flags
var (
	flagNumIterations = flag.Int("num_iterations", 0, "Number of iterations of self-play and then train. "+
		"A value of <= 0 means to train indefinitely, until interrupted.")
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

	// Capture Control+C
	var globalCancel func()
	globalCtx, globalCancel = context.WithCancel(context.Background())
	spinning.SafeInterrupt(globalCancel, 5*time.Second)
	defer globalCancel()

	// Profilers: HTTP profiler server and CPU profile.
	profilers.Setup(globalCtx)
	defer profilers.OnQuit()

	// Create AI player.
	must.M(createAIPlayer())
	defer func() {
		// Clean up -- before stopping at the profiler, if it is enabled.
		aiPlayer = nil
		bootstrapAiPlayer = nil
	}()

	// Iterate over playing and training.
	if *flagNumIterations <= 0 {
		fmt.Println("Training indefinitely (use -num_iterations to limit it):")
		fmt.Println("\t- It saves after each iteration, and you can simply interrupt (Control+C) when you want to stop.")
	}
	var currentExamples []Example
	for i := 0; *flagNumIterations <= 0 || i < *flagNumIterations; i++ {
		fmt.Printf("\nIteration #%d - %s\n", i, time.Now().Format("2006-01-02 15:04:05"))
		var newExamples []Example
		if bootstrapAiPlayer != nil {
			_, _, _, newExamples = must.M4(runMatches(globalCtx, *flagNumMatches, aiPlayer, bootstrapAiPlayer))
		} else {
			_, _, _, newExamples = must.M4(runMatches(globalCtx, *flagNumMatches, aiPlayer, aiPlayer))
		}
		if globalCtx.Err() != nil {
			// Interrupted.
			return
		}
		if currentExamples == nil {
			currentExamples = newExamples
		} else {
			currentExamples = append(currentExamples, newExamples...)
		}
		if success := must.M1(trainAI(globalCtx, currentExamples)); success {
			// AI was replaced by the improved one, discard examples before generating new ones.
			currentExamples = currentExamples[:0]
		}
		if globalCtx.Err() != nil {
			return
		}
	}
}
