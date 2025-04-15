// Package profilers implement helper functions to set up profiling for the various programs.
//
// If linked, it will install the profiler flags.
//
// It only supports debugging, and otherwise has no functionality for the Hive game.
package profilers

import (
	"context"
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

	// globalCtx is set on the call to Setup.
	globalCtx context.Context
)

// Setup starts the HTTP (flag -prof) and CPU profilers (flag -cpu_profile), if they were configured.
// You should follow with a deferred call to OnQuit.
func Setup(ctx context.Context) {
	globalCtx = ctx
	if *flagProfiler >= 0 {
		setupHTTPProfiler()
	}
	if *flagCPUProfile != "" {
		createCPUProfile()
	}
}

// OnQuit should be called before the exit of the main() function, typically this is setup as a deferred call
// just after Setup.
func OnQuit() {
	if *flagCPUProfile != "" {
		pprof.StopCPUProfile()
	}
	if *flagProfiler >= 0 {
		httpProfilerOnQuit()
	}
}

// createCPUProfile creates the file pointed by *flagCPUProfile and starts the CPU profiling there.
// It returns the function to be called on stop.
func createCPUProfile() {
	f, err := os.Create(*flagCPUProfile)
	if err != nil {
		klog.Fatal("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		klog.Fatal("could not start CPU profile: ", err)
	}
	return
}

// setupHTTPProfiler starts the profiler if it was enabled by the -prof flag.
// If it was not enabled, it is a no-op.
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
//
// If the profiler was not enabled (-prof), it is a No-op.
func httpProfilerOnQuit() {
	if *flagProfiler <= 0 {
		return
	}
	// Don't freeze on panic.
	if err := recover(); err != nil {
		panic(err)
	}
	if globalCtx.Err() != nil {
		// Already interrupted.
		return
	}

	// Garbage collect, to see if there is anything leaking.
	for _ = range 10 {
		runtime.GC()
	}
	fmt.Printf("- Program finished: kept alive with profiler opened at %s/debug/pprof\n", profilerAddr)
	fmt.Printf("- Interrupt (Ctrl+C) to exit\n")
	<-globalCtx.Done()
	fmt.Printf("... exiting ...\n")
}
