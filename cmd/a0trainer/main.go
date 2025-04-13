package main

import (
	"context"
	"flag"
	"fmt"
	_ "github.com/janpfeifer/hiveGo/internal/players/default"
	"github.com/janpfeifer/hiveGo/internal/ui/spinning"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"time"
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

// main orchestrates playing, loading, rescoring, saving and training of matches.
func main() {
	klog.InitFlags(nil)
	flag.Parse()

	// Capture Control+C
	var globalCancel func()
	globalCtx, globalCancel = context.WithCancel(context.Background())
	spinning.SafeInterrupt(globalCancel, 5*time.Second)
	defer globalCancel()

	// Profilers: HTTP profiler server and CPU profile.
	if *flagProfiler >= 0 {
		setupHTTPProfiler()
		defer httpProfilerOnQuit()
	}
	if *flagCPUProfile != "" {
		stopCPUProfile := createCPUProfile()
		defer stopCPUProfile()
	}

	// Create AI player.
	must.M(createAIPlayer())

	// Iterate over playing and training.
	//var examples []Example
	newExamples := must.M1(runMatches(globalCtx))
	fmt.Printf("Example 0: %+v\n", newExamples[0])
	fmt.Printf("Example 1: %+v\n", newExamples[1])
}
