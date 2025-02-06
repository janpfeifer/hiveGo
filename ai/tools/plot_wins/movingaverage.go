package main

import (
	"maps"
	"slices"
)

// MaxMovingAverageWeight carried over from past examples.
const MaxMovingAverageWeight = 0.999

// MatchResult represents either one player wins or a draw.
type MatchResult int

const (
	P0Wins MatchResult = iota
	P1Wins
	Draw
)

// MovingAverages for the 3 possible match results.
// They should sum up to 1.0 (except at the start where they are 0).
type MovingAverages [3]float32

// addResult moving average with a new result.
func (ma *MovingAverages) addResult(result MatchResult, count int) {
	weight := 1.0 - 1.0/float32(count)
	if weight > MaxMovingAverageWeight {
		weight = MaxMovingAverageWeight
	}
	for possibleResult := range Draw + 1 {
		ma[possibleResult] = ma[possibleResult] * weight
		if possibleResult == result {
			ma[possibleResult] += 1.0 - weight
		}
	}
}

// IndexedMatchResult includes the match index, and the result of the match.
type IndexedMatchResult struct {
	Index  int
	Result MatchResult
}

// WinsMovingAverage will read individual match results and output the current moving average.
//
// It will hold on to the stats if results come out-of-order, reporting
// only when results on the correct order is available.
//
// The matchResults should be fed starting from 0. It handles restart of the sequence, for instance
// if a new training session is started (by resetting the moving average), but the restart has to start
// with index 0 again, and it's not perfect if results from two sequences are interleaved.
// An index of -1 can be used as a flag to  trigger a start of a new sequence, and the result is actually ignored.
//
// It only returns when matchResults is closed, so it should be run on a separate goroutine.
// At exit, it closes statsChan.
func WinsMovingAverage(matchResults <-chan IndexedMatchResult, statsChan chan<- MovingAverages) {
	defer close(statsChan)

	// nextMatchIndex we want to use.
	nextMatchIndex := 0
	var stats MovingAverages
	pending := make(map[int]MatchResult)
	count := 0
	for mr := range matchResults {
		if mr.Index < nextMatchIndex {
			// Restarting from 0 if a new sequence of matches is fed: it's not perfect
			// if results from two sequences are interleaved.
			// Reset moving average
			count = flushPending(count, stats, pending, statsChan)
			nextMatchIndex = 0
			// Optionally it could reset the stats, since a new
			// training session started. But, if the training session has the
			// same parameters, it makes more sense to continue with the
			// moving average.
			// ? Maybe add a flag for stats = [3]float32{0, 0, 0}
			clear(pending)
			if mr.Index < 0 {
				continue
			}
		}

		if mr.Index > nextMatchIndex {
			// Cache result for future use.
			pending[mr.Index] = mr.Result
			continue
		}

		// MatchResult in order, add the result and yield it.
		nextMatchIndex++
		count++
		stats.addResult(mr.Result, count)
		statsChan <- stats

		// Flush any cached following results.
		for {
			result, found := pending[nextMatchIndex]
			if !found {
				break
			}
			delete(pending, nextMatchIndex)
			nextMatchIndex++
			count++
			stats.addResult(result, count)
			statsChan <- stats
			//fmt.Printf("Next: %d\n", nextMatchIndex)
		}
	}
	count = flushPending(count, stats, pending, statsChan)
	close(statsChan)
}

// flushPending results from a sequence in order, even if not all results are in.
func flushPending(count int, stats MovingAverages, pending map[int]MatchResult, statsChan chan<- MovingAverages) int {
	sortedKeys := slices.Collect(maps.Keys(pending))
	slices.Sort(sortedKeys)
	for _, key := range sortedKeys {
		result := pending[key]
		count++
		stats.addResult(result, count)
		statsChan <- stats
	}
	return count
}
