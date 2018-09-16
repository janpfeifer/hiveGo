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

	"github.com/golang/glog"
	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf

// TrivialScorer is a linear model (one weight per feature + bias)
// on the feature set.
type LinearScorer []float64

func (w LinearScorer) UnlimitedScore(features []float64) float64 {
	// Sum start with bias.
	sum := w[len(w)-1]

	// Dot product of weights and features.
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
	features := FeatureVector(b)
	sum := w.UnlimitedScore(features)
	if sum > 9.8 {
		sum = 9.8
	} else if sum < -9.8 {
		sum = -9.8
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
	parts := make([]string, len(w)+2*(len(AllFeatures)+1))
	for _, fDef := range AllFeatures {
		parts = append(parts, fmt.Sprintf("\n\t// %s -> %d\n\t", fDef.Name, fDef.Dim))
		for _, value := range w[fDef.VecIndex : fDef.VecIndex+fDef.Dim] {
			parts = append(parts, fmt.Sprintf("%.4f, ", value))
		}
		parts = append(parts, "\n")
	}
	parts = append(parts, fmt.Sprintf("\n\t// Bias -> 1\n\t"))
	parts = append(parts, fmt.Sprintf("%.4f,\n", w[len(w)-1]))
	return strings.Join(parts, "")
}

var (
	cacheLinearScorers = map[string]LinearScorer{}
	muLinearModels     sync.Mutex
)

func (w LinearScorer) Learn(learningRate float64, examples []LabeledExample) float64 {
	//log.Printf("  w_t  =%s\n", w)
	grad := make([]float64, AllFeaturesDim+1)
	totalLoss := 0.0
	for _, example := range examples {
		// Loss = Sqr(label - score)
		// dLoss/dw_i = 2*(label-score)*x_i
		// dLoss/b = 2*(label-score)
		score := w.UnlimitedScore(example.Features)
		loss := example.Label - score
		loss = loss * loss
		totalLoss += loss
		c := learningRate * 2 * (example.Label - score)
		for ii, feature := range example.Features {
			grad[ii] += c * feature
		}
		grad[len(grad)-1] += c
	}
	totalLoss /= float64(len(examples))
	totalLoss = math.Sqrt(totalLoss)
	//log.Printf("Root Mean Squared Loss=%.2f", totalLoss)

	// Sum gradient and regularization.
	for ii := range grad {
		grad[ii] /= float64(len(examples))
	}
	clip(grad, 0.1)
	for ii := range grad {
		w[ii] += grad[ii] // 1e-2*w[ii]
	}
	//log.Printf("  w_t+1=%s\n", w)
	return totalLoss
}

func length(vec []float64) float64 {
	total := 0.0
	for _, value := range vec {
		total += value * value
	}
	return math.Sqrt(total)
}

func clip(vec []float64, max float64) {
	l := length(vec)
	if l > max {
		ratio := max / l
		for ii := range vec {
			vec[ii] *= ratio
		}
	}
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
		glog.Infof("Using cache for model '%s'", file)
		return cached
	}
	defer func() { cacheLinearScorers[file] = w }()

	w = make(LinearScorer, len(TrainedBest))
	_, err := os.Stat(file)
	if os.IsNotExist(err) {
		// Make fresh copy of TrainedBest
		copy(w, TrainedBest)
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
	TrainedV0 = LinearScorer{
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// F_NUM_OFFBOARD
		-0.43, 0.04, -0.52, -2.02, -0.64,
		// F_OPP_NUM_OFFBOARD
		0.53, 0.29, 0.41, 1.71, 0.73,

		// F_NUM_SURROUNDING_QUEEN / F_OPP_NUM_SURROUNDING_QUEEN
		-3.15, 3.86,

		// F_NUM_CAN_MOVE
		0.75, 0.0, 0.57, 0.0, -0.07, 0.0, 1.14, 0.0, 0.07, 0.0,
		// F_OPP_NUM_CAN_MOVE
		0.05, 0.0, -0.17, 0.0, 0.13, 0.0, 0.26, 0.0, 0.02, 0.0,

		// F_NUM_THREATENING_MOVES
		0., 0.,

		// F_NUM_TO_DRAW
		0.,

		// Bias: *Must always be last*
		-0.79,
	}

	TrainedV1 = LinearScorer{
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// F_NUM_OFFBOARD
		0.04304, 0.03418, 0.04503, -1.863, 0.0392,
		// F_OPP_NUM_OFFBOARD
		0.05537, 0.03768, 0.04703, 1.868, 0.03902,

		// F_NUM_SURROUNDING_QUEEN / F_OPP_NUM_SURROUNDING_QUEEN
		-3.112, 3.422,

		// F_NUM_CAN_MOVE
		0.6302, 0.0, 0.4997, 0.0, -0.1359, 0.0, 1.115, 0.0, 0.0436, 0.0,

		// F_OPP_NUM_CAN_MOVE
		-0.001016, 0.0, -0.2178, 0.0, 0.05738, 0.0, 0.2827, 0.0, -0.01102, 0.0,

		// F_NUM_THREATENING_MOVES
		-0.1299, -0.04499,

		// F_NUM_TO_DRAW
		0.00944,

		// Bias: *Must always be last*
		-0.8161,
	}

	TrainedV2 = LinearScorer{
		// NumOffboard -> 5
		0.0440, 0.0346, 0.0369, -1.8634, 0.0369,

		// OppNumOffboard -> 5
		0.0547, 0.0380, 0.0539, 1.8684, 0.0408,

		// NumSurroundingQueen -> 1
		-3.0499,

		// OppNumSurroundingQueen -> 1
		3.3732,

		// NumCanMove -> 10
		0.6028, 0.0095, 0.4872, -0.0047, -0.1144, 0.0167, 1.0972, -0.0178, 0.0450, 0.0037,

		// OppNumCanMove -> 10
		0.0147, -0.0115, -0.2071, 0.0026, 0.0468, -0.0128, 0.2932, 0.0105, -0.0029, 0.0006,

		// NumThreateningMoves -> 2
		-0.0976, 0.0070,

		// MovesToDraw -> 1
		0.0031,

		// Bias -> 1
		-0.8164,
	}

	TrainedBest = TrainedV2
)
