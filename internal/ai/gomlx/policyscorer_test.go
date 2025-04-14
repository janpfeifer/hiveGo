package gomlx

import (
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/stretchr/testify/require"
	"runtime"
	"testing"
)

func TestPolicyScorer_Clone(t *testing.T) {
	model := NewAlphaZeroFNN()
	params := parameters.NewFromConfigString("a0fnn,max_traverses=100")
	s0, err := newPolicyScorer(ModelAlphaZeroFNN, "", model, params)
	require.NoError(t, err)
	b := state.NewBoard()
	require.NotPanics(t, func() { s0.PolicyScore(b) })

	learner, err := s0.CloneLearner()
	require.NoError(t, err)
	s1, ok := learner.(*PolicyScorer)
	require.True(t, ok)

	// Both scorers working.
	require.NotPanics(t, func() { s0.PolicyScore(b) })
	require.NotPanics(t, func() { s1.PolicyScore(b) })

	// s0 is finalized, and s1 must still work.
	s0.Finalize()
	for _ = range 10 {
		runtime.GC()
	}
	require.NotPanics(t, func() { s1.PolicyScore(b) })
}
