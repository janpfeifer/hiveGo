// Package spinning provides a friendly spinning clock (or some other spinning symbols)
// to use while the a program is calculating something.
package spinning

import (
	"context"
	"fmt"
	"k8s.io/klog/v2"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

type Spinning struct {
	wg     sync.WaitGroup
	cancel func()
}

var (
	ThemeAscii = []rune("|/-\\")
	ThemeMoon  = []rune("🌑🌒🌓🌔🌕🌖🌗🌘")
	ThemeClock = []rune("🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛")

	// Theme defaults to ThemeClock, but it can be set to anything else.
	Theme       = ThemeClock
	spinningIdx int
	themeLen    = len(Theme)
)

// SafeInterrupt will capture SigInt (Ctrl+C) and SigTerm and call the provided onInterrupt.
// If the program haven't exited after gracePeriod, it will call Reset to reset the terminal
// and exit.
func SafeInterrupt(onInterrupt func(), gracePeriod time.Duration) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		s := <-sigChan
		fmt.Println()
		klog.Errorf("Got interrupted (signal %q), shutting down... (%s)", s, gracePeriod)
		if onInterrupt != nil {
			go onInterrupt()
		}

		// Wait for gracePeriod before exiting.
		time.Sleep(gracePeriod)
		Reset()
		klog.Fatalf("Graceful shutting down %s period expired, exiting.", gracePeriod)
	}()
}

// Reset terminal: make cursor visible, restore default terminal colors.
func Reset() {
	fmt.Print("\033[?25h\033[39;49;0m\n") // Restore cursor and colors.
}

// New starts a spinning display that runs on a separate GoRoutine.
// It stops when Spinning.Done is called.
func New(ctx context.Context) *Spinning {
	s := &Spinning{}
	ctx, s.cancel = context.WithCancel(ctx)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		ticker := time.NewTicker(500 * time.Millisecond)
		fmt.Print("\033[?25l")       // Hide cursor.
		defer fmt.Print("\033[?25h") // Restore cursor.

		fmt.Print("  ")
		for {
			symbol := Theme[spinningIdx]
			fmt.Printf("\b\b%c", symbol)
			spinningIdx = (spinningIdx + 1) % themeLen
			select {
			case <-ctx.Done():
				fmt.Print("\b\b")
				return
			case <-ticker.C:
				// continue
			}
		}
	}()
	return s
}

func (s *Spinning) Done() {
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil
	}
	s.wg.Wait()
}
