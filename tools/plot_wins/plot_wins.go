package main

import (
	"bufio"
	"fmt"
	"github.com/janpfeifer/hiveGo/tools/wins_ma"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

func main() {
	fmt.Println(`"P0 Wins" "P1 Wins" "Draw"`)

	// Setup pipeline to process results.
	resultsChan := make(chan wins_ma.MatchResult)
	statsChan := make(chan wins_ma.Stats)
	var wg sync.WaitGroup
	go wins_ma.WinsMovingAverage(resultsChan, statsChan)
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
			`Match (\d+) finished \(`+
				`(player [01]|draw)`)
		resetString = `Created TensorFlow device`
	)

	// Scan lines.
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, resetString) {
			resultsChan <- wins_ma.MatchResult{-1, wins_ma.DRAW }
		}
		matches := reResult.FindStringSubmatch(line)
		if matches == nil {
			continue
		}
		idx, err := strconv.Atoi(matches[1])
		if err != nil {
			log.Fatal("Failed to parse match num %v", err)
		}
		var r wins_ma.Result
		if matches[2] == "player 0" {
			r = wins_ma.P0_WINS
		} else if matches[2] == "player 1" {
			r = wins_ma.P1_WINS
		} else if matches[2] == "draw" {
			r = wins_ma.DRAW
		} else {
			log.Fatal("Unparseable match result: %s", line)
		}
		resultsChan <- wins_ma.MatchResult{idx, r}
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}
