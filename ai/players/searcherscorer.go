package players

import (
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ai/search"
	"github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
)

// SearcherScorer is a standard set up for an AI: a searcher and a scorer.
// It implements the Player interface.
type SearcherScorer struct {
	Searcher     search.Searcher
	Scorer       ai.BatchBoardScorer
	Learner      ai.LearnerScorer
	ModelPath    string
	Parallelized bool
}

// Play implements the Player interface: it chooses an action given a Board.
func (p *SearcherScorer) Play(b *state.Board, matchName string) (
	action state.Action, board *state.Board, score float32, actionsLabels []float32) {
	action, board, score, actionsLabels = p.Searcher.Search(b)
	klog.V(1).Infof("Match %s, Move #%d (%s): AI playing %v, score=%.3f",
		matchName, b.MoveNumber, p.ModelPath, action, score)
	return
}
