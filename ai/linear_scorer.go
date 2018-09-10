package ai

import (
	"log"

	. "github.com/janpfeifer/hiveGo/state"
)

var _ = log.Printf

// TrivialScorer is a linear model (one weight per feature + bias)
// on the feature set.
type LinearScorer []float64

func (w LinearScorer) Score(b *Board) float64 {
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

var (
	ManualV0 = BatchScorerWrapper{LinearScorer{
		// Pieces order: ANT, BEETLE, GRASSHOPPER, QUEEN, SPIDER
		// F_NUM_OFFBOARD
		-0.1, -0.05, -0.03, -0.4, -0.02,
		// F_OPP_NUM_OFFBOARD
		0.1, 0.05, 0.03, 0.4, 0.02,
		// F_NUM_SURROUNDING_QUEEN / F_OPP_NUM_SURROUNDING_QUEEN
		-0.5, 0.5,
		// F_NUM_CAN_MOVE
		0.2, 0.1, 0.08, 0.2, 0.08,
		// F_OPP_NUM_CAN_MOVE
		-0.2, -0.1, -0.08, -0.2, -0.08,
		// Bias
		0.0,
	}}
)
