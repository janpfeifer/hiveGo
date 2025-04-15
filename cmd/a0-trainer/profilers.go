package main

import (
	"flag"
	"fmt"
	"k8s.io/klog/v2"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
)

var (
	flagProfiler   = flag.Int("prof", -1, "If set, runs the profile at the given port.")
	flagCPUProfile = flag.String("cpu_profile", "", "write cpu profile to `file`")
	profilerAddr   string
)

// createCPUProfile creates the file pointed by *flagCPUProfile and starts the CPU profiling there.
// It returns the function to be called on stop.
func createCPUProfile() func() {
	f, err := os.Create(*flagCPUProfile)
	if err != nil {
		klog.Fatal("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		klog.Fatal("could not start CPU profile: ", err)
	}
	return pprof.StopCPUProfile
}

// setupHTTPProfiler starts the profiler.
func setupHTTPProfiler() {
	profilerAddr = fmt.Sprintf("localhost:%d", *flagProfiler)
	fmt.Printf("Starting profiler on %s/debug/pprof\n", profilerAddr)
	fmt.Printf("- You can access it with: $ go tool pprof %s/debug/pprof/heap\n", profilerAddr)
	fmt.Printf("- Program will be kept alive on end, you will have to interrupt it (Ctrl+C) to exit\n")
	go func() {
		klog.Fatal(http.ListenAndServe(profilerAddr, nil))
	}()
	// Freeze while not interrupted, so one can read profile.
}

// httpProfilerOnQuit is a deferred function called on exit if the profiler is configured:
// it keeps the program alive until interrupt is called.
func httpProfilerOnQuit() {
	// Don't freeze on panic.
	if err := recover(); err != nil {
		panic(err)
	}
	if globalCtx.Err() != nil {
		// Already interrupted.
		return
	}

	// Free AI player resources and garbage collect, to see if there is anything
	// leaking.
	aiPlayer = nil
	bootstrapAiPlayer = nil
	for _ = range 10 {
		runtime.GC()
	}
	fmt.Printf("- Program finished: kept alive with profiler opened at %s/debug/pprof\n", profilerAddr)
	fmt.Printf("- Interrupt (Ctrl+C) to exit\n")
	<-globalCtx.Done()
	fmt.Printf("... exiting ...\n")
}
