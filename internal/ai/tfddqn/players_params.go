package tfddqn

import (
	players2 "github.com/janpfeifer/hiveGo/internal/players"
	"log"
)

// Data used for parsing of player options.
type ParsingData struct {
	Model     string
	UseTFDDQN bool
}

func NewParsingData() (data interface{}) {
	return &ParsingData{}
}

func FinalizeParsing(data interface{}, player *players2.SearcherScorer) {
	d := data.(*ParsingData)
	if d.UseTFDDQN {
		player.Learner = New(d.Model)
		player.Scorer = player.Learner
		player.ModelPath = d.Model
	}
}

func ParsePlayerParam(data interface{}, key, value string) {
	d := data.(*ParsingData)
	if key == "tfddqn" {
		d.UseTFDDQN = true
	} else if key == "model" {
		d.Model = value
	} else {
		log.Panicf("Unknown parameter '%s=%s' passed to tensorflow module.", key, value)
	}
}

func init() {
	for _, key := range []string{"model", "tfddqn"} {
		players2.RegisterModule(
			"tfddqn", key, NewParsingData, ParsePlayerParam, FinalizeParsing,
			players2.ScorerType)
	}
}
