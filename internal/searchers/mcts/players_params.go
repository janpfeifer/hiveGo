package mcts

import (
	players2 "github.com/janpfeifer/hiveGo/internal/players"
	"log"
	"time"
)

func init() {
	for _, key := range []string{"mcts", "c_puct", "max_time", "max_traverses",
		"min_traverses", "max_score", "max_depth", "randomness"} {
		players2.RegisterModule("mcts", key,
			NewParsingData, ParsePlayerParam, FinalizeParsing,
			players2.SearcherType)
	}
}

func NewParsingData() (data interface{}) {
	return &mctsSearcher{
		maxDepth:     10,
		maxTime:      30 * time.Second,
		maxTraverses: 300,
		minTraverses: 0,
		maxAbsScore:  9.0,
		cPuct:        1.1,
		randomness:   0.0,
		parallelized: false,
		useMCTS:      false,
	}
}

func ParsePlayerParam(data interface{}, key, value string) {
	d := data.(*mctsSearcher)
	if key == "mcts" {
		d.useMCTS = true
	} else if key == "c_puct" {
		d.cPuct = players2.MustFloat32(value, key)
		if d.cPuct < 0 {
			log.Panicf("Negative c_puct value not possible")
		}
	} else if key == "max_time" {
		sec := players2.MustInt(value, key)
		d.maxTime = time.Second * time.Duration(sec)
	} else if key == "max_depth" {
		d.maxDepth = players2.MustInt(value, key)
	} else if key == "max_traverses" {
		d.maxTraverses = players2.MustInt(value, key)
	} else if key == "min_traverses" {
		d.minTraverses = players2.MustInt(value, key)
	} else if key == "max_score" {
		d.maxAbsScore = players2.MustFloat32(value, key)
	} else if key == "randomness" {
		d.randomness = players2.MustFloat32(value, key)
	} else {
		log.Panicf("Unknown parameter '%s=%s' passed to mcts module.", key, value)
	}
}

func FinalizeParsing(data interface{}, player *players2.SearcherScorer) {
	d := data.(*mctsSearcher)
	if d.useMCTS {
		if player.Searcher != nil {
			log.Panicf("Searcher already selected while setting up MCTS.")
		}
		if player.Scorer == nil {
			log.Panicf("MCTS requires a scorer.")
		}
		d.scorer = player.Scorer
		player.Searcher = d
	}
}
