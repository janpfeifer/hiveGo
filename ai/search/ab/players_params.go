package ab

import (
	"github.com/golang/klog.
	"github.com/janpfeifer/hiveGo/ai/players"
	"log"
)

func init() {
	for _, key := range []string{"ab", "max_depth", "randomness"} {
		players.RegisterPlayerParameter("ab", key,
			NewParsingData, ParsePlayerParam, FinalizeParsing,
			players.SearcherType)
	}
}

func NewParsingData() (data interface{}) {
	return &alphaBetaSearcher{
		maxDepth:     2,
		randomness:   0.0,
		parallelized: false,
	}
}

func ParsePlayerParam(data interface{}, key, value string) {
	d := data.(*alphaBetaSearcher)
	if key == "ab" {
		d.useAB = true
	} else if key == "max_depth" {
		d.maxDepth = players.MustInt(value, key)
	} else if key == "randomness" {
		d.randomness = players.MustFloat32(value, key)
	} else {
		log.Panicf("Unknown parameter '%s=%s' passed to ab module.", key, value)
	}
}

func FinalizeParsing(data interface{}, player *players.SearcherScorerPlayer) {
	d := data.(*alphaBetaSearcher)
	if d.useAB {
		klog.V(1).Info("Creating AlphaBetaPruning searcher")
		if player.Searcher != nil {
			log.Panicf(
				"Searcher already selected while setting up AlphaBetaPruning.")
		}
		if player.Scorer == nil {
			log.Panicf("AlphaBetaPruning requires a scorer.")
		}
		d.scorer = player.Scorer
		player.Searcher = d
	}
}
