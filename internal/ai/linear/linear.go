// Package linear implements a pure Go linear scorer that can be used to play as well as
// training -- it defines its own gradient for that, and can be used for a simple SGD.
package linear

import (
	"bytes"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/features"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"log"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// Scorer is a linear model (one weight per feature + bias) on the feature set.
// It implements ai.BoardScorer.
type Scorer struct {
	name string

	weights []float32

	// LearningRate to use when training the linear model and L2Reg to use.
	LearningRate, L2Reg float32

	// GradientL2Clip clips the gradient to this l2 length before applying.
	GradientL2Clip float32

	// Linearize training.
	muLearning sync.Mutex

	// batchSize recommended for calling Learn.
	batchSize int

	// FileName where to save/load the model from.
	FileName string
	muSave   sync.Mutex
}

// NewWithWeights creates a new Scorer with the given weights.
// Ownership of the weights is transferred.
func NewWithWeights(weights ...float32) *Scorer {
	return &Scorer{
		name:           "custom",
		weights:        weights,
		LearningRate:   0.001,
		L2Reg:          1e-4,
		GradientL2Clip: 1.0,
		batchSize:      100,
	}
}

var (
	// Assert Scorer is an ai.BoardScorer and an ai.BatchBoardScorer
	_ ai.BoardScorer      = (*Scorer)(nil)
	_ ai.BatchBoardScorer = (*Scorer)(nil)
	_ ai.LearnerScorer    = (*Scorer)(nil)
)

// String implements fmt.Stringer and ai.Scorer.
func (s *Scorer) String() string {
	var fileNameDesc string
	if s.FileName != "" {
		fileNameDesc = " (path=" + s.FileName + ")"
	}
	return fmt.Sprintf("Linear %q%s", s.name, fileNameDesc)
}

// Clone creates a deep copy of the Scorer.
//
// Notice s must not be locked.
func (s *Scorer) Clone() *Scorer {
	s2 := &Scorer{}
	//goland:noinspection ALL
	*s2 = *s
	s2.weights = slices.Clone(s.weights)
	return s2
}

// WithName sets the name of the Scorer and returns itself.
func (s *Scorer) WithName(name string) *Scorer {
	s.name = name
	return s
}

// WithBatchSize sets the recommended batch size.
func (s *Scorer) WithBatchSize(batchSize int) *Scorer {
	s.batchSize = batchSize
	return s
}

// WithLearningRate sets the learning rate for the Scorer and returns itself.
func (s *Scorer) WithLearningRate(learningRate float32) *Scorer {
	s.LearningRate = learningRate
	return s
}

func (s *Scorer) logitScore(features []float32) float32 {
	// Sum start with bias.
	sum := s.weights[len(s.weights)-1]

	// Dot product of weights and features.
	if len(s.weights)-1 != len(features) {
		log.Panicf("Features dimension is %d, but weights dimension is %d (+1 bias)",
			len(features), len(s.weights)-1)
	}
	for ii, feature := range features {
		sum += feature * s.weights[ii]
	}
	return sum
}

// BoardScore implements ai.BoardScorer.
func (s *Scorer) BoardScore(board *Board) float32 {
	return s.ScoreFeatures(features.ForBoard(board, s.Version()))
}

// ScoreFeatures is like Score, but it takes the raw features as input.
func (s *Scorer) ScoreFeatures(rawFeatures []float32) float32 {
	logit := s.logitScore(rawFeatures)
	return ai.SquashScore(logit)
}

// BatchBoardScore implements ai.BatchBoardScorer.
func (s *Scorer) BatchBoardScore(boards []*Board) (scores []float32) {
	scores = make([]float32, len(boards))
	for ii, board := range boards {
		scores[ii] = s.BoardScore(board)
	}
	return
}

// AsGoCode outputs the model as Go code describing the weights for each feature.
func (s *Scorer) AsGoCode() string {
	if len(s.weights) != features.BoardFeaturesDim+1 {
		return fmt.Sprintf("model with %d weights+1 bias, BoardFeaturesDim=%d", len(s.weights)-1, features.BoardFeaturesDim)
	}
	buf := bytes.NewBuffer(make([]byte, 0, 16*1024))
	for _, fDef := range features.BoardSpecs {
		_, _ = fmt.Fprintf(buf, "\t// %s -> [%d]\n\t", fDef.Id, fDef.Dim)
		for _, value := range s.weights[fDef.VecIndex : fDef.VecIndex+fDef.Dim] {
			_, _ = fmt.Fprintf(buf, "%.4f, ", value)
		}
		_, _ = fmt.Fprintf(buf, "\n\n")
	}
	_, _ = fmt.Fprintf(buf, "\t// Bias -> [1]\n\t%.4f,\n", s.weights[len(s.weights)-1])
	return buf.String()
}

// l2RegularizationLoss is the regularization term for the loss.
func (s *Scorer) l2RegularizationLoss() float32 {
	if s.L2Reg == 0 {
		return 0
	}
	sum := float32(0)
	for _, param := range s.weights {
		sum += param * param
	}
	return sum * s.L2Reg
}

// Learn implements ai.LearnerScorer, and trains model with the new boards and its labels.
// It returns the loss.
func (s *Scorer) Learn(boards []*Board, boardLabels []float32) (loss float32) {
	//fmt.Printf("Learn(%d boards)\n", len(boards))
	s.muLearning.Lock()
	defer s.muLearning.Unlock()

	// Build features.
	boardFeatures := make([][]float32, len(boards))
	for boardIdx, board := range boards {
		boardFeatures[boardIdx] = features.ForBoard(board, s.Version())
	}
	return s.lockedLearnFromFeatures(boardFeatures, boardLabels)
}

// lockedLearnFromFeatures uses as input a batch of feature vectors (as opposed to the raw boards)
// It assumes s.muLearning is already locked.
func (s *Scorer) lockedLearnFromFeatures(boardFeatures [][]float32, boardLabels []float32) (loss float32) {
	// Loop over steps.
	grad := make([]float32, len(s.weights))
	s.calculateGradient(boardFeatures, boardLabels, grad)

	// Clip gradient.
	if s.GradientL2Clip > 0 {
		clipL2(grad, s.GradientL2Clip)
	}
	//fmt.Printf("\tL2(grad) = %.4f\n", l2Len(grad))

	// Apply gradient with the learning rate.
	for ii := range grad {
		//wasZero := grad[ii] == 0
		//signBefore := math32.Signbit(s.weights[ii])
		s.weights[ii] -= s.LearningRate * grad[ii]
		//if !wasZero && math32.Signbit(s.weights[ii]) != signBefore {
		//	// When crossing the 0 barrier, make the gradient stop at 0 first.
		//	s.weights[ii] = 0
		//}
	}
	return s.lossFromFeatures(boardFeatures, boardLabels)
}

// calculateGradient of the MSE (MeanSquaredError) loss:
//
//	  x, x_i: input (features) and x term i
//	  w, w_i: weights, and weight term i
//	  b: bias term of the model
//	  score: tanh(w*x+b)
//	Loss = (label - score)^2/N
//	  dLoss/dw_i = (2*(score-label)*d(score)/dw_i)/N
//	  dLoss/db = (2*(score-label)*d(score)/db)/N
//	  d(score)/dw_i = (1-score^2)*x_i
//	  d(score)/db = (1-score^2)
func (s *Scorer) calculateGradient(inputs [][]float32, labels []float32, gradient []float32) {
	for i := range gradient {
		gradient[i] = 0
	}
	N := float32(len(inputs))
	for exampleIdx, x := range inputs {
		score := s.ScoreFeatures(x)
		c := 2 * (score - labels[exampleIdx]) * (1 - score*score)
		for i, x_i := range x {
			// dLoss/dw_i
			gradient[i] += c * x_i
		}
		// gradient of the bias term (the last)
		gradient[len(gradient)-1] += c
	}

	// Take the mean:
	for ii := range gradient {
		gradient[ii] /= N
	}
	if s.L2Reg > 0 {
		// L2 regularization
		for ii := range s.weights {
			gradient[ii] += 2 * s.weights[ii] * s.L2Reg
		}
	}
}

// Loss returns the loss of the model given the labels.
func (s *Scorer) Loss(boards []*Board, boardLabels []float32) (loss float32) {
	// Build features.
	boardFeatures := make([][]float32, len(boards))
	for boardIdx, board := range boards {
		boardFeatures[boardIdx] = features.ForBoard(board, s.Version())
	}
	return s.lossFromFeatures(boardFeatures, boardLabels)
}

// lossFromFeatures calculates the loss after the boards have been converted to features (x).
func (s *Scorer) lossFromFeatures(boardFeatures [][]float32, boardLabels []float32) (loss float32) {
	for boardIdx, x := range boardFeatures {
		// MSE (MeanSquaredLoss):
		//     x, x_i: input (features) and x term i
		//     w, w_i: weights, and weight term i
		//     b: bias term of the model
		//     score: tanh(w*x+b)
		//   Loss = (label - score)^2/N + L2Reg(w) + L2Reg(b)
		score := s.ScoreFeatures(x)
		diff := boardLabels[boardIdx] - score
		loss += diff * diff
	}
	N := float32(len(boardLabels))
	loss /= N
	loss += s.l2RegularizationLoss()
	return
}

// BatchSize returns the recommended batch size and implements ai.LearnerScorer.
func (s *Scorer) BatchSize() int {
	return s.batchSize
}

func l2Len(vec []float32) float32 {
	total := float32(0.0)
	for _, value := range vec {
		total += value * value
	}
	return float32(math.Sqrt(float64(total)))
}

// clipL2 clips the L2 length of the vector.
func clipL2(vec []float32, maxLen float32) {
	l2 := l2Len(vec)
	if l2 > maxLen {
		ratio := maxLen / l2
		//fmt.Printf("\tclip: l2=%g, maxLen=%g, ratio=%g\n", l2, maxLen, ratio)
		for ii := range vec {
			vec[ii] *= ratio
		}
		//fmt.Printf("\tvec=%v\n", vec)
	}
}

// Version of the features this model operates: the convention is that the version is the same as the number of features.
func (s *Scorer) Version() int {
	return len(s.weights) - 1
}

// Save model to s.FileName.
// It implements ai.LearnerScorer.
func (s *Scorer) Save() error {
	s.muSave.Lock()
	defer s.muSave.Unlock()

	if s.FileName == "" {
		klog.Errorf("Linear model not saved, because no file name was specified")
		return nil
	}

	// Rename existing file, if it exists.
	file := s.FileName
	if _, err := os.Stat(file); err == nil {
		err = os.Rename(file, file+"~")
		if err != nil {
			return errors.Wrapf(err, "failed to rename %s to %s", s.FileName, s.FileName+"~")
		}
	} else if !os.IsNotExist(err) {
		return errors.Wrapf(err, "failed to stat %s", s.FileName)
	}

	err := os.WriteFile(s.FileName, []byte(s.AsGoCode()), 0777)
	if err != nil {
		return errors.Wrapf(err, "failed to save %s", s.FileName)
	}
	return nil
}

// Cache of linear models read from disk.
var (
	cacheLinearScorers = map[string]*Scorer{}
	muCache            sync.Mutex
)

// LoadOrCreate model from fileName or create a new one, bootstrapped from PreTrainedBest.
// It stores the reference of the loaded or created model in a cache, that is reused if attempting to load
// the same fileName.
func LoadOrCreate(fileName string, base *Scorer) (*Scorer, error) {
	if fileName == "" {
		return PreTrainedBest, nil
	}

	muCache.Lock()
	defer muCache.Unlock()
	if cached, ok := cacheLinearScorers[fileName]; ok {
		klog.V(1).Infof("Using cache for model %q", fileName)
		return cached, nil
	}

	_, err := os.Stat(fileName)
	if os.IsNotExist(err) {
		// Make fresh copy of PreTrainedBest
		var weights []float32
		if base == nil {
			klog.Infof("Creating zero-initialized new linear model in %s using %d features", fileName, features.BoardFeaturesDim)
			weights = make([]float32, features.BoardFeaturesDim+1)
		} else {
			klog.Infof("Creating new linear model in %s copied from %s", fileName, base.name)
			weights = slices.Clone(base.weights)
		}
		s := NewWithWeights(weights...).WithName(fileName)
		s.FileName = fileName
		cacheLinearScorers[fileName] = s
		return s, nil
	}

	data, err := os.ReadFile(fileName)
	if err != nil {
		return nil, errors.Wrapf(err, "LoadOrCreate failed to read file %s", fileName)
	}
	lines := strings.Split(string(data), "\n")
	weights := make([]float32, 0, len(lines))
	for lineNum, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
			// Skip empty lines and comments.
			continue
		}
		values := strings.Split(line, ",")
		for _, value := range values {
			value := strings.TrimSpace(value)
			if len(value) == 0 {
				continue
			}
			f64, err := strconv.ParseFloat(value, 32)
			if err != nil {
				return nil, errors.Wrapf(err, "LoadOrCreate failed to parse value %q in file %s, at line number #%d",
					value, fileName, lineNum+1)
			}
			weights = append(weights, float32(f64))
		}
	}
	s := NewWithWeights(weights...)
	s.FileName = fileName
	cacheLinearScorers[fileName] = s
	return s, nil
}
