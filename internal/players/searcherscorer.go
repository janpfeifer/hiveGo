package players

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/janpfeifer/hiveGo/internal/searchers"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"strings"
)

// SearcherScorer is a standard set up for an AI: a searcher and a scorer.
// It implements the Player interface.
type SearcherScorer struct {
	Searcher       searchers.Searcher
	PolicySearcher searchers.SearcherWithPolicy

	ValueScorer   ai.ValueScorer
	ValueLearner  ai.ValueLearner
	PolicyScorer  ai.PolicyScorer
	PolicyLearner ai.PolicyLearner
}

// New creates a new AI player given the configuration string.
//
// Args:
//
//   - config: a comma-separated list of parameters with optional values associated. At least a "scorer" (e.g. "linear")
//     and a "searcher" (e.g. "ab" or "mcts") must be defined. If empty, the default is given by DefaultPlayerConfig.
//     E.g.: "linear,ab,max_depth=2"
//
// Typical parameters:
//
//   - linear (string): Configure to use the linear scorer. Default value is "best", and other valid values are
//     "v0", "v1", "v2" or a path to the linear model to be loaded.
//   - ab (bool): If to use Alpha-Beta pruning search algorithm.
//   - mcts (bool): Use MCTS (Monte Carlo Tree Search) algorithm.
//   - max_depth (int): Max depth of search, default is 2. If max_time is set, this parameter is ignored.
//   - max_time (time.Duration): Max time duration in search, default is 0s, which means it is not time-limited but rather max_depth limited.
//   - randomness (float): Adds a layer of randomness in the search: the first level choice is
//     distributed according to a softmax of the scores of each move, divided by this value.
//     So lower values (closer to 0) means less randomness, higher value means more randomness,
//     hence more exploration. Default is 0.
//
// More details on the config are dependent on the module used.
func New(config string) (*SearcherScorer, error) {
	if config == "" {
		config = DefaultPlayerConfig
	}
	params := parameters.NewFromConfigString(config)
	player := &SearcherScorer{}

	if len(RegisteredScorers) == 0 {
		return nil, errors.New("no registered scorers. Perhaps you need to import _ \"github.com/janpfeifer/hiveGo/internal/player/default\" to your binary ?")
	}
	if len(RegisteredSearchers) == 0 {
		return nil, errors.New("no registered searchers. Perhaps you need to import _ \"github.com/janpfeifer/hiveGo/internal/player/default\" to your binary ?")
	}

	// Find scorer.
	for _, builder := range RegisteredScorers {
		s, err := builder(params)
		if err != nil {
			return nil, err
		}
		if s == nil {
			// Not this type of scorer.
			continue
		}
		if player.ValueScorer != nil {
			return nil, errors.Errorf("multiple scorers defined in parameters %q", config)
		}
		player.ValueScorer = s
	}
	if player.ValueScorer == nil {
		return nil, errors.Errorf("no scorers defined in parameters %q", config)
	}

	// Find searcher.
	for _, builder := range RegisteredSearchers {
		s, err := builder(player.ValueScorer, params)
		if err != nil {
			return nil, err
		}
		if s == nil {
			continue
		}
		if player.Searcher != nil {
			return nil, errors.Errorf("multiple searchers defined in parameters %q", config)
		}
		player.Searcher = s
	}
	if player.Searcher == nil {
		return nil, errors.Errorf("no searchers defined in parameters %q", config)
	}
	if policySearcher, ok := player.Searcher.(searchers.SearcherWithPolicy); ok {
		player.PolicySearcher = policySearcher
	}

	// Check whether the scorer is also a learner.
	if learner, ok := player.ValueScorer.(ai.ValueLearner); ok {
		player.ValueLearner = learner
	}
	if policyScorer, ok := player.ValueScorer.(ai.PolicyScorer); ok {
		player.PolicyScorer = policyScorer
	}
	if policyLearner, ok := player.ValueScorer.(ai.PolicyLearner); ok {
		player.PolicyLearner = policyLearner
	}

	// Check that all parameters were processed.
	if len(params) > 0 {
		return nil, errors.Errorf("unknown AI parameters \"%s\" passed", strings.Join(generics.KeysSlice(params), "\", \""))
	}
	return player, nil
}

// Assert that SearchScorer is a Player.
var _ Player = &SearcherScorer{}

// Play implements the Player interface: it chooses an action given a Board.
func (s *SearcherScorer) Play(b *Board) (
	action Action, board *Board, score float32, actionsLabels []float32) {
	var err error
	if s.PolicySearcher != nil {
		action, board, score, actionsLabels, err = s.PolicySearcher.SearchWithPolicy(b)
	} else {
		action, board, score, err = s.Searcher.Search(b)
	}
	if err != nil {
		klog.Fatal("Searcher error: %+v", err)
	}
	if klog.V(2).Enabled() {
		klog.Infof("Move #%d: AI (%s) playing %s, score=%.3f",
			b.MoveNumber, s.ValueScorer, action, score)
	}
	return
}

func (s *SearcherScorer) String() string {
	return fmt.Sprintf("%s (searcher=%s)", s.ValueScorer, s.Searcher)
}

// Finalize is called at the end of a match.
func (s *SearcherScorer) Finalize() {
	if klog.V(1).Enabled() {
		klog.Infof("Player (scorer=%s) finalized", s.ValueScorer)
	}
	s.ValueScorer = nil
	s.Searcher = nil
}
