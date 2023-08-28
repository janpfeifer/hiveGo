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

const MAX_LINEAR_SCORE = float32(9.8)

// TrivialScorer is a linear model (one weight per feature + bias)
// on the feature set.
type LinearScorer []float32

func (w LinearScorer) UnlimitedScore(features []float32) float32 {
	// Sum start with bias.
	sum := w[len(w)-1]

	// Dot product of weights and features.
	if len(w)-1 != len(features) {
		log.Panicf("Features dimension is %d, but weights dimension is %d (+1 bias)",
			len(features), len(w)-1)
	}
	for ii, feature := range features {
		sum += feature * w[ii]
	}
	return sum
}

// Adjusts numbers larger than 10 to approximate 10 in the infinity, by applying
// a sigmoid to anything above 9.8 -- in absolute terms, it works symmetrically
// on negative numbers.
func SigmoidTo10(x float32) float32 {
	if x < MAX_LINEAR_SCORE && x > -MAX_LINEAR_SCORE {
		return x
	}
	sign := float32(1)
	abs := x
	if x < 0 {
		sign = -1
		abs = -x
	}

	// Calculate sigmoid part.
	const reduction = float32(4) // Makes it converge slower to 10.0
	sig := (abs - MAX_LINEAR_SCORE) / reduction
	sig = float32(1.0 / (1.0 + math.Exp(-float64(sig))))
	sig = (sig - 0.5) * 2 * (END_GAME_SCORE - MAX_LINEAR_SCORE)
	abs = 9.8 + sig
	return sign * abs
}

func (w LinearScorer) ScoreFeatures(features []float32) float32 {
	return SigmoidTo10(w.UnlimitedScore(features))
}

func (w LinearScorer) Score(b *Board, scoreActions bool) (score float32, actionProbs []float32) {
	if scoreActions {
		glog.Error("LinearScorer.Score() doesn't support scoreActions.")
	}
	features := FeatureVector(b, w.Version())
	return SigmoidTo10(w.UnlimitedScore(features)), nil
}

func (w LinearScorer) BatchScore(boards []*Board, scoreActions bool) (scores []float32, actionProbsBatch [][]float32) {
	if scoreActions {
		glog.Error("LinearScorer.BatchScore() doesn't support scoreActions.")
	}
	scores = make([]float32, len(boards))
	actionProbsBatch = nil
	for ii, board := range boards {
		scores[ii], _ = w.Score(board, scoreActions)
	}
	return
}

func (w LinearScorer) String() string {
	if len(w) != AllFeaturesDim+1 {
		return fmt.Sprintf("Model with %d features, AllFeaturesDim=%d", len(w)-1, AllFeaturesDim)
	}
	parts := make([]string, len(w)+2*(len(w)))
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

func (w LinearScorer) Learn(boards []*Board, boardLabels []float32,
	actionsLabels [][]float32, learningRate float32, steps int,
	perStepCallback func()) (loss, boardLoss, actionsLoss float32) {
	// Bulid features.
	boardFeatures := make([][]float32, len(boards))
	for boardIdx, board := range boards {
		boardFeatures[boardIdx] = FeatureVector(board, w.Version())
	}

	var totalLoss float32
	for step := 0; step < steps || step == 0; step++ {
		grad := make([]float32, len(w))
		totalLoss = 0
		for boardIdx, features := range boardFeatures {
			// Loss = Sqr(label - score)
			// dLoss/dw_i = 2*(label-score)*x_i
			// dLoss/b = 2*(label-score)
			score := w.UnlimitedScore(features)
			loss := boardLabels[boardIdx] - score
			loss = loss * loss
			totalLoss += loss
			c := learningRate * 2 * (boardLabels[boardIdx] - score)
			for ii, feature := range features {
				grad[ii] += c * feature
			}
			grad[len(grad)-1] += c
		}
		totalLoss /= float32(len(boards))
		totalLoss = float32(math.Sqrt(float64(totalLoss)))

		// Sum gradient and regularization.
		if step < steps {
			for ii := range grad {
				grad[ii] /= float32(len(boards))
			}
			clip(grad, 0.1)
			for ii := range grad {
				w[ii] += grad[ii]
			}
			if perStepCallback != nil {
				perStepCallback()
			}
		}
	}
	return totalLoss, totalLoss, 0
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

func (w LinearScorer) IsActionsClassifier() bool {
	return false
}

func (w LinearScorer) Version() int {
	return len(w) - 1
}

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

	if file == "" {
		return TrainedBest
	}

	if cached, ok := cacheLinearScorers[file]; ok {
		glog.Infof("Using cache for model '%s'", file)
		return cached
	}
	defer func() { cacheLinearScorers[file] = w }()

	_, err := os.Stat(file)
	if os.IsNotExist(err) {
		// Make fresh copy of TrainedBest
		w = make(LinearScorer, AllFeaturesDim+1)
		if TrainedBest.Version() != AllFeaturesDim {
			glog.Errorf("New model with %d features initialized with current best with %d, it may not make much sense ?", w.Version(), TrainedBest.Version())
		}
		copy(w, TrainedBest)
		glog.V(1).Infof("New model has %d features", w.Version())
		return
	}

	data, err := ioutil.ReadFile(file)
	check(err)
	valuesStr := strings.Split(string(data), "\n")
	w = make(LinearScorer, len(valuesStr))

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

		// F_NUM_SINGLE,
		0., 0.,

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

		// F_OPP_NUM_THREATENING_MOVES
		// 0.1299, 0.04499,

		// F_NUM_TO_DRAW
		0.00944,

		// F_NUM_SINGLE,
		0., 0.,

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

		// F_NUM_SINGLE,
		0., 0.,

		// Bias -> 1
		-0.8147,
	}

	TrainedV3 = LinearScorer{
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// NumOffboard -> 5
		-0.1891, 0.0648, -0.0803, -1.8169, -0.0319,

		// OppNumOffboard -> 5
		0.2967, 0.0625, 0.2306, 2.2038, 0.1646,

		// NumSurroundingQueen -> 1
		-2.4521,

		// OppNumSurroundingQueen -> 1
		2.7604,

		// NumCanMove -> 10
		0.0065, 0.4391, 0.5519, -0.4833, -0.0156, 0.1361, 0.9591, -0.1592, 0.2405, 0.0343,

		// OppNumCanMove -> 10
		0.2158, -0.7518, -0.4865, 0.3333, 0.1677, -0.2764, -0.0134, -0.2725, 0.0145, 0.0184,

		// NumThreateningMoves -> 2
		0.0087, 0.0714,

		// OppNumThreateningMoves -> 2
		0.1979, 0.0486,

		// MovesToDraw -> 1
		-0.0074,

		// NumSingle -> 2
		-0.2442, 0.3755,

		// QueenIsCovered -> 2
		0, 0, // -10, 10,

		// AverageDistanceToQueen
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		0, -0.01, -0.01, 0, -0.01,

		// OppAverageDistanceToQueen
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		0, 0.001, 0.001, 0, 0.001,

		// Bias -> 1
		-0.7409,
	}

	TrainedBest = TrainedV3
)
