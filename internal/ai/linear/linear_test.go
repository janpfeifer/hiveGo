package linear

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"math/rand/v2"
	"testing"
)

func TestScoreAndGradient(t *testing.T) {
	model := NewWithWeights(
		// Weights:
		2, -1, 4,
		// Bias:
		3)
	model.L2Reg = 0
	model.GradientL2Clip = 0

	grad := make([]float32, len(model.weights))
	for ii, test := range []struct {
		inputs             []float32
		label, score, loss float32
		grad               []float32
	}{
		// Examples generated using GoMLX:
		{inputs: []float32{0, 0, 0}, label: 0.9, score: 0.995055, loss: 0.009035,
			grad: []float32{0, 0, 0, 0.001876}},
		{inputs: []float32{0.1, 0.1, 0.1}, label: 0.9, score: 0.998178, loss: 0.009639,
			grad: []float32{7.14887e-05, 7.14887e-05, 7.14887e-05, 0.000715}},
		{inputs: []float32{-0.9, -0.9, -0.9}, label: 0.9, score: -0.905148, loss: 3.258560,
			grad: []float32{0.58716404, 0.58716404, 0.58716404, -0.652404}},
		// score=0.905148, loss=0.000027, gradW=[], gradB=
		{inputs: []float32{0.2, -0.1, -0.5}, label: 0.9, score: 0.905148, loss: 0.000027,
			grad: []float32{0.00037213214, -0.00018606607, -0.00093033037, 0.001861}},
	} {
		score := model.ScoreFeatures(test.inputs)
		loss := model.lossFromFeatures([][]float32{test.inputs}, []float32{test.label})
		model.calculateGradient([][]float32{test.inputs}, []float32{test.label}, grad)
		fmt.Printf("#%d: x=%v, score=%f, label=%f, loss=%v, grad=%v\n", ii, test.inputs, score, test.label, loss, grad)
		assert.InDelta(t, test.score, score, 1e-4)
		assert.InDelta(t, test.loss, loss, 1e-4)
		assert.InDeltaSlice(t, test.grad, grad, 1e-4)
	}
}

// TestLearn checks that is is able to learn our "want" linear model from examples.
func TestLearn(t *testing.T) {
	want := NewWithWeights(
		// Weights:
		2, -1, 4,
		// Bias:
		3)

	// The model being trained:
	got := NewWithWeights(0, 0, 0, 0)
	got.LearningRate = 0.01
	got.L2Reg = 0

	rng := rand.New(rand.NewPCG(42, 0)) // Ensure reproducibility
	// Generating a single large blob of float32 for all examples
	numExamples := 10_000
	numFeatures := 3
	dataBlob := make([]float32, numExamples*numFeatures)
	for i := range dataBlob {
		dataBlob[i] = float32(rng.NormFloat64()) // Mean 0, Std 1
	}

	// Creating subslices for each example
	examples := make([][]float32, numExamples)
	for i := 0; i < numExamples; i++ {
		examples[i] = dataBlob[i*numFeatures : (i+1)*numFeatures]
	}

	// Generating labels using the "want" model
	labels := make([]float32, numExamples)
	for i := 0; i < numExamples; i++ {
		labels[i] = want.ScoreFeatures(examples[i])
	}

	// Training `got` using lockedLearnFromFeatures
	batchSize := 100
	numEpochs := 1000
	var finalLoss float32
	for epoch := 0; epoch < numEpochs; epoch++ {
		for i := 0; i < numExamples; i += batchSize {
			end := i + batchSize
			if end > numExamples {
				end = numExamples
			}
			batchExamples := examples[i:end]
			batchLabels := labels[i:end]
			finalLoss = got.lockedLearnFromFeatures(batchExamples, batchLabels)
			fmt.Printf("\repoch %2d: loss: %.3g      ", epoch, finalLoss)
			//fmt.Printf("got: %v\n", got.weights)
		}
	}
	fmt.Println()
	fmt.Printf("     want: %v\n", want.weights)
	fmt.Printf("final got: %v\n", got.weights)

	// Assertions
	assert.Less(t, finalLoss, float32(1e-4))
	assert.InDeltaSlice(t, want.weights, got.weights, 0.4)
}
