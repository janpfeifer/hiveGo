package mcts

import (
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	. "github.com/janpfeifer/hiveGo/internal/state"
	. "github.com/janpfeifer/hiveGo/internal/state/statetest"
	"github.com/stretchr/testify/require"
	"testing"
)

// dummyScorer returns a 0 value score for all boards, and equal probability policy for all actions.
type dummyScorer struct{}

func (s *dummyScorer) PolicyScore(board *Board) []float32 {
	numActions := board.NumActions()
	policy := make([]float32, numActions)
	for ii := range policy {
		policy[ii] = 1.0 / float32(numActions)
	}
	return policy
}

func (s *dummyScorer) Score(board *Board) float32 {
	return 0
}

func (s *dummyScorer) String() string {
	return "dummyScorer"
}

var (
	_ ai.ValueScorer  = &dummyScorer{}
	_ ai.PolicyScorer = &dummyScorer{}
)

func buildTestMCTS(t *testing.T, config string) *Searcher {
	params := parameters.NewFromConfigString(config)
	searcher, err := NewFromParams(&dummyScorer{}, params)
	require.NoError(t, err)
	return searcher.(*Searcher)
}

func TestMctsSearcher_Search_EndGame(t *testing.T) {
	layout := []PieceOnBoard{
		{Pos{0, 0}, 0, ANT},
		{Pos{-1, 0}, 1, BEETLE},
		{Pos{1, 0}, 0, SPIDER},
		{Pos{-1, 1}, 1, GRASSHOPPER},
		{Pos{2, 0}, 0, QUEEN},
		{Pos{-1, 2}, 1, QUEEN},
		{Pos{1, 1}, 0, SPIDER},
		{Pos{2, 1}, 1, ANT},
		{Pos{3, 0}, 0, GRASSHOPPER},
		{Pos{-2, 0}, 1, ANT},
		{Pos{3, -1}, 0, BEETLE},
	}
	board := BuildBoard(layout, false)
	board.NextPlayer = 1
	board.BuildDerived()
	PrintBoard(board)

	mcts := buildTestMCTS(t, "mcts,max_traverses=2000,min_traverses=10,temperature=0")
	winningAction := Action{Move: true, Piece: ANT, SourcePos: Pos{-2, 0}, TargetPos: Pos{2, -1}}
	actionTaken, _, score, policy, err := mcts.SearchWithPolicy(board)
	require.NoError(t, err)
	PrintActions(board, actionTaken, policy)
	actionIdx := -1
	for idx, action := range board.Derived.Actions {
		if action == actionTaken {
			actionIdx = idx
			break
		}
	}
	require.Greater(t, actionIdx, -1)

	require.Equal(t, board.NumActions(), len(policy))
	var totalProb float32
	for _, prob := range policy {
		totalProb += prob
	}
	require.InDelta(t, float32(1), totalProb, 1e-4)
	require.Equal(t, winningAction, actionTaken)
	require.Greater(t, score, float32(0.95))
	require.Greater(t, policy[actionIdx], float32(0.95))
}
