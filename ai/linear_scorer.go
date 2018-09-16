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
type LinearScorer []float32

func (w LinearScorer) UnlimitedScore(features []float32) float32 {
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

func (w LinearScorer) Score(b *Board) float32 {
	features := FeatureVector(b)
	sum := w.UnlimitedScore(features)
	if sum > 9.8 {
		sum = 9.8
	} else if sum < -9.8 {
		sum = -9.8
	}
	return sum
}

func (w LinearScorer) BatchScore(boards []*Board) []float32 {
	scores := make([]float32, len(boards))
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

func (w LinearScorer) Learn(learningRate float32, examples []LabeledExample, steps int) float32 {
	var totalLoss float32
	for step := 0; step < steps; step++ {
		grad := make([]float32, AllFeaturesDim+1)
		totalLoss = 0
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
		totalLoss /= float32(len(examples))
		totalLoss = float32(math.Sqrt(float64(totalLoss)))

		// Sum gradient and regularization.
		for ii := range grad {
			grad[ii] /= float32(len(examples))
		}
		clip(grad, 0.1)
		for ii := range grad {
			w[ii] += grad[ii] // 1e-2*w[ii]
		}
	}
	return totalLoss
}

func length(vec []float32) float32 {
	total := float32(0.0)
	for _, value := range vec {
		total += value * value
	}
	return float32(math.Sqrt(float64(total)))
}

func clip(vec []float32, max float32) {
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

// Horrible hack, but ... adding the file name to the LinearScorer object
// would take lots of refactoring.
var LinearModelFileName string

func (w LinearScorer) Save() {
	muLinearModels.Lock()
	defer muLinearModels.Unlock()

	file := LinearModelFileName
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
		f64, err := strconv.ParseFloat(valuesStr[ii], 32)
		w[ii] = float32(f64)
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
		0.0446, 0.0340, 0.0287, -1.8639, 0.0321,

		// OppNumOffboard -> 5
		0.0532, 0.0373, 0.0572, 1.8688, 0.0432,

		// NumSurroundingQueen -> 1
		-3.0338,

		// OppNumSurroundingQueen -> 1
		3.3681,

		// NumCanMove -> 10
		0.5989, 0.0090, 0.4845, -0.0045, -0.1100, 0.0213, 1.0952, -0.0198, 0.0468, 0.0054,

		// OppNumCanMove -> 10
		0.0185, -0.0096, -0.2016, 0.0043, 0.0483, -0.0133, 0.2939, 0.0113, 0.0004, 0.0037,

		// NumThreateningMoves -> 2
		-0.0946, 0.0114,

		// MovesToDraw -> 1
		0.0033,

		// Bias -> 1
		-0.8147,
	}

	TrainedBest = TrainedV2
)
