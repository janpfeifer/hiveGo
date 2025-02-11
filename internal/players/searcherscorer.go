package players

import (
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"github.com/janpfeifer/hiveGo/internal/searchers"
	"github.com/janpfeifer/hiveGo/internal/searchers/alphabeta"
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
//   - max_time (time.Duration): Max time duration in search, default is 0s, which means it is not time-limited but rather max_depth limited.
//   - randomness (float): Adds a layer of randomness in the search: the first level choice is
//     distributed according to a softmax of the scores of each move, divided by this value.
//     So lower values (closer to 0) means less randomness, higher value means more randomness,
//     hence more exploration. Default is 0.
//
// It returns a SearcherScorer, which implements the Player interface, but also has support for online learning.
func NewPlayerFromScorer(scorer ai.BoardScorer, matchId uint64, matchName string, playerNum PlayerNum, params map[string]string) (*SearcherScorer, error) {
	// Shared parameters.
	player := &SearcherScorer{
		matchId:   matchId,
		matchName: matchName,
		playerNum: playerNum,
	}

	// Batch scorer:
	batchScorer, ok := scorer.(ai.BatchBoardScorer)
	if !ok {
		batchScorer = &ai.BatchBoardScorerWrapper{scorer}
	}
	player.Scorer = batchScorer

	// Searcher:
	isMCTS, err := PopParamOr(params, "mcts", false)
	if err != nil {
		return nil, err
	}
	isAB, err := PopParamOr(params, "ab", !isMCTS)
	if err != nil {
		return nil, err
	}
	if isAB && isMCTS {
		return nil, errors.New("you can only choose one of \"ab\" (alpha-beta pruning) or " +
			"\"mcts\" (Monte Carlo Tree Search) searcher")
	}
	if isAB {
		player.Searcher = alphabeta.New(batchScorer)
	} else {
		return nil, errors.New("MCTS not connected yet.")
	}

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
	matchName string
	playerNum PlayerNum

	Searcher searchers.Searcher
	Scorer   ai.BatchBoardScorer
	Learner  ai.LearnerScorer
}

// Assert that SearchScorer is a Player.
var _ Player = &SearcherScorer{}

// Play implements the Player interface: it chooses an action given a Board.
func (s *SearcherScorer) Play(b *Board) (
	action Action, board *Board, score float32, actionsLabels []float32) {
	action, board, score, actionsLabels = s.Searcher.Search(b)
	if klog.V(1).Enabled() {
		klog.Infof("Match %q (#%d), %s player, Move #%d (%s): AI playing %v, score=%.3f",
			s.matchName, s.matchId, s.playerNum, b.MoveNumber, action, score)
	}
	return
}

// Finalize is called at the end of a match.
func (s *SearcherScorer) Finalize() {
	if klog.V(1).Enabled() {
		klog.Infof("Player for match %q (#%d), %s player finalized", s.matchName, s.matchId, s.playerNum)
	}
	s.Scorer = nil
	s.Searcher = nil
}
