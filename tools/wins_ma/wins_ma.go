// wins_ma is a package that implements a moving average of
// players' wins and draws.
//
// Since matches are played in parallel and end asynchronously,
// this library also
package wins_ma

import "sort"

// The moving average will carry at most that much weight from
// past examples.
const MAX_MOVING_AVERGAGE_WEIGHT = 0.99

type Result int

const (
	P0_WINS Result = iota
	P1_WINS
	DRAW
)

// Stats is a moving average for the 3 conditions of the match. They should
// sum up to 1.0.
type Stats [3]float32

func (s *Stats) CombineResult(r Result, count int) {
	weight := 1.0 - 1.0/float32(count)
	if weight > MAX_MOVING_AVERGAGE_WEIGHT {
		weight = MAX_MOVING_AVERGAGE_WEIGHT
	}
	for ii := Result(0); ii <= DRAW; ii++ {
		s[ii] = s[ii] * weight
		if ii == r {
			s[ii] += (1.0 - weight)
		}
	}
}

// MatchResult includes the match index, and the result of the match.
type MatchResult struct {
	Index  int
	Result Result
}

// WinsMovingAverage will read individual match results and output the
// current moving average.
//
// It will hold on to the stats if results come out-of-order, reporting
// only when results on the correct order is available.
//
// It will close statsChan and return when matchResults is closed.
func WinsMovingAverage(matchResults <-chan MatchResult, statsChan chan<- Stats) {
	next := 0
	var stats Stats
	pending := make(map[int]Result)
	for mr := range matchResults {
		//fmt.Printf("mr=%v\n", mr)
		// Starting if starting a new sequence.
		if mr.Index < next {
			// Reset moving average
			flushPending(stats, pending, next, statsChan)
			next = 0
			stats = [3]float32{0, 0, 0}
			pending = make(map[int]Result)
			if mr.Index < 0 {
				continue
			}
		}

		if mr.Index > next {
			// Store result for future use.
			pending[mr.Index] = mr.Result
			//fmt.Printf("Storing: %d, next: %d\n", mr.Index, next)
		} else {
			// Result in order, combine it.
			next++
			stats.CombineResult(mr.Result, next)
			statsChan <- stats
			//fmt.Printf("Next: %d\n", next)

			// Check if next results are already available.
			for {
				r, found := pending[next]
				if !found {
					break
				}
				delete(pending, next)
				next++
				stats.CombineResult(r, next)
				statsChan <- stats
				//fmt.Printf("Next: %d\n", next)
			}
		}
	}
	flushPending(stats, pending, next, statsChan)
	close(statsChan)
}

func flushPending(stats Stats, pending map[int]Result, next int, statsChan chan<- Stats) {
	keys := make([]int, 0, len(pending))
	for key := range pending {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	for _, key := range keys {
		r := pending[key]
		next++
		stats.CombineResult(r, next)
		statsChan <- stats
	}
}
