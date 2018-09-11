package ai

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf

// TrivialScorer is a linear model (one weight per feature + bias)
// on the feature set.
type LinearScorer []float64

func (w LinearScorer) UnlimitedScore(b *Board) float64 {
	// Sum start with bias.
	sum := w[len(w)-1]

	// Dot product of weights and features.
	features := FeatureVector(b)
	if len(w)-1 != len(features) {
		log.Fatalf("Features dimension is %d, but weights dimension is %d (+1 bias)",
			len(features), len(w)-1)
	}
	for ii, feature := range features {
		sum += feature * w[ii]
	}
	return sum
}

func (w LinearScorer) Score(b *Board) float64 {
	sum := w.UnlimitedScore(b)
	if math.Abs(sum) > 9.8 {
		sum /= math.Abs(sum)
		sum *= 9.8
	}
	return sum
}

func (w LinearScorer) BatchScore(boards []*Board) []float64 {
	scores := make([]float64, len(boards))
	for ii, board := range boards {
		scores[ii] = w.Score(board)
	}
	return scores
}

func (w LinearScorer) String() string {
	parts := make([]string, len(w))
	for ii, value := range w {
		parts[ii] = fmt.Sprintf("%.2f", value)
	}
	return fmt.Sprintf("[%s]", strings.Join(parts, ", "))
}

var (
	cacheLinearScorers = map[string]LinearScorer{}
	muLinearModels     sync.Mutex
)

func (w LinearScorer) Learn(b *Board, label float64) {
	// Loss = Sqr(label - score)
	// dLoss/dw_i = 2*(label-score)*x_i
	// dLoss/b = 2*(label-score)
	score := w.UnlimitedScore(b)
	features := FeatureVector(b)

	log.Printf("Features: %v\n", features)
	log.Printf("  score=%.2f, label=%.2f\n", score, label)
	log.Printf("  w_t  =%s\n", w)

	muLinearModels.Lock()
	defer muLinearModels.Unlock()

	const learningRate = 0.005
	c := learningRate * 2 * (label - score)
	for ii, feature := range features {
		dw := c * feature
		if math.Abs(dw) > 1.0 {
			dw /= math.Abs(dw)
		}
		w[ii] += dw
	}
	db := c
	if math.Abs(db) >= 1.0 {
		db = db / math.Abs(db)
	}
	w[len(w)-1] += c

	// Regularization.
	for ii := range w {
		w[ii] -= 1e-5 * w[ii]
	}

	log.Printf("  w_t+1=%s\n", w)
}

func check(err error) {
	if err != nil {
		log.Panicf("Failed: %v", err)
	}
}

func (w LinearScorer) Save(file string) {
	muLinearModels.Lock()
	defer muLinearModels.Unlock()

	if _, err := os.Stat(file); err == nil {
		err = os.Rename(file, file+"~")
		if err != nil {
			log.Printf("Failed to rename '%s' to '%s~': %v", file, file, err)
		}
	}

	valuesStr := make([]string, len(w))
	for ii, value := range w {
		valuesStr[ii] = fmt.Sprintf("%g", value)
	}
	allValues := strings.Join(valuesStr, "\n")

	err := ioutil.WriteFile(file, []byte(allValues), 0777)
	check(err)
}

func NewLinearScorerFromFile(file string) (w LinearScorer) {
	muLinearModels.Lock()
	defer muLinearModels.Unlock()

	if cached, ok := cacheLinearScorers[file]; ok {
		log.Printf("Using cache for model '%s'", file)
		return cached
	}
	defer func() { cacheLinearScorers[file] = w }()

	w = make(LinearScorer, len(ManualV0))
	_, err := os.Stat(file)
	if os.IsNotExist(err) {
		// Make fresh copy of ManualV0
		copy(w, ManualV0)
		return
	}

	data, err := ioutil.ReadFile(file)
	check(err)
	valuesStr := strings.Split(string(data), "\n")
	if len(valuesStr) < len(w) {
		log.Fatalf("Model file '%s' only has %d values, need %d",
			len(valuesStr), len(w))
	}
	for ii := 0; ii < len(w); ii++ {
		w[ii], err = strconv.ParseFloat(valuesStr[ii], 64)
		check(err)
	}
	return
}

var (
	// Values actually trained with LinearScorer.Learn.
	ManualV0 = LinearScorer{
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// F_NUM_OFFBOARD
		-0.43, 0.04, -0.52, -2.02, -0.64,
		// F_OPP_NUM_OFFBOARD
		0.53, 0.29, 0.41, 1.71, 0.73,

		// F_NUM_SURROUNDING_QUEEN / F_OPP_NUM_SURROUNDING_QUEEN
		-3.15, 3.86,

		// F_NUM_CAN_MOVE
		0.75, 0.57, -0.07, 1.14, 0.07,
		// F_OPP_NUM_CAN_MOVE
		0.05, -0.17, 0.13, 0.26, 0.02,

		// Bias
		-0.79,
	}
)
