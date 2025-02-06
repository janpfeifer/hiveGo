package main

import (
	"bufio"
	"fmt"
	"k8s.io/klog/v2"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

func main() {
	fmt.Println(`"P0 Wins" "P1 Wins" "Draw"`)

	// Setup pipeline to process results.
	resultsChan := make(chan IndexedMatchResult)
	statsChan := make(chan MovingAverages)
	var wg sync.WaitGroup
	go WinsMovingAverage(resultsChan, statsChan)
	wg.Add(1)
	go func() {
		for stats := range statsChan {
			fmt.Printf("%g %g %g\n",
				stats[0], stats[1], stats[2])
		}
		wg.Done()
	}()
	defer func() {
		// Wait for last results to come out.
		close(resultsChan)
		wg.Wait()
	}()

	var (
		reResult = regexp.MustCompile(
			`Match (\d+) finished \(` +
				`(player [01]|draw)`)
		resetString = `Created TensorFlow device`
	)

	// Scan lines.
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, resetString) {
			resultsChan <- IndexedMatchResult{-1, Draw}
		}
		matches := reResult.FindStringSubmatch(line)
		if matches == nil {
			continue
		}
		idx, err := strconv.Atoi(matches[1])
		if err != nil {
			klog.Fatalf("Failed to parse match num %v", err)
		}
		var r MatchResult
		if matches[2] == "player 0" {
			r = P0Wins
		} else if matches[2] == "player 1" {
			r = P1Wins
		} else if matches[2] == "draw" {
			r = Draw
		} else {
			klog.Fatalf("Unparseable match result: %s", line)
		}
		resultsChan <- IndexedMatchResult{idx, r}
	}
	if err := scanner.Err(); err != nil {
		klog.Fatalf("Failed reading stdin: %+v", err)
	}
}
