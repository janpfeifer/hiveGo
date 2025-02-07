package players

import (
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"github.com/janpfeifer/hiveGo/internal/search"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"strings"
)

// NewPlayerFromScorer creates a new AI player given a BoardScorer, that does alpha-beta-pruning or MCTS (Monte Carlo Tress Search).
// If the scorer is an ai.BatchBoardScorer, that is used instead.
//
// Params:
//
//   - ab (bool): If to use Alpha-Beta pruning search algorithm. This is the default.
//   - mcts (bool): Use MCTS (Monte Carlo Tree Search) algorith, the default is Alpha-Beta Pruning.
//   - max_depth (int): Max depth of search, default is 2. If max_time is set, this parameter is ignored.
//   - max_time (time.Duration): Max time duration in search, default is 0s, which means it is not time limited but rather max_depth limited.
//   - randomness (float): Adds a layer of randomness in the search: the first level choice is
//     distributed according to a softmax of the scores of each move, divided by this value.
//     So lower values (closer to 0) means less randomness, higher value means more randomness,
//     hence more exploration. Default is 0.
//
// It returns a SearcherScorer, which implements the Player interface, but also has support for online learning.
func NewPlayerFromScorer(scorer ai.BoardScorer, matchId uint64, playerNum PlayerNum, params map[string]string) (*SearcherScorer, error) {

	// Shared parameters.
	player := &SearcherScorer{
		matchId:   matchId,
		playerNum: playerNum,
	}

	// Batch scorer:
	batchScorer, ok := scorer.(ai.BatchBoardScorer)
	if !ok {
		batchScorer = &ai.BatchBoardScorerWrapper{scorer}
	}
	player.Scorer = batchScorer

	// Searcher:

	// Check that all parameters were processed.
	if len(params) > 0 {
		return nil, errors.Errorf("unknown AI parameters \"%s\" passed", strings.Join(generics.KeysSlice(params), "\", \""))
	}
	return player, nil
}

// SearcherScorer is a standard set up for an AI: a searcher and a scorer.
// It implements the Player interface.
type SearcherScorer struct {
	matchId   uint64
	playerNum PlayerNum

	Searcher search.Searcher
	Scorer   ai.BatchBoardScorer
	Learner  ai.LearnerScorer
}

// Assert that SearchScorer is a Player.
var _ Player = &SearcherScorer{}

// Play implements the Player interface: it chooses an action given a Board.
func (p *SearcherScorer) Play(b *Board, matchName string) (
	action Action, board *Board, score float32, actionsLabels []float32) {
	action, board, score, actionsLabels = p.Searcher.Search(b)
	klog.V(1).Infof("Match %s, Move #%d (%s): AI playing %v, score=%.3f",
		matchName, b.MoveNumber, p.ModelPath, action, score)
	return
}

// Finalize is called at the end of a match.
func (p *SearcherScorer) Finalize() {
	p.Scorer = nil
	p.Searcher = nil
}
