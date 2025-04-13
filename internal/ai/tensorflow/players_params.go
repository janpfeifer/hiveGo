package tensorflow

import (
	players2 "github.com/janpfeifer/hiveGo/internal/players"
	"log"
	"strconv"
)

// Data used for parsing of player options.
type ParsingData struct {
	Model                   string
	UseTensorFlow, ForceCPU bool
	SessionPoolSize         int
}

func NewParsingData() (data interface{}) {
	return &ParsingData{SessionPoolSize: 1}
}

func FinalizeParsing(data interface{}, player *players2.SearcherScorer) {
	d := data.(*ParsingData)
	if d.UseTensorFlow {
		player.ValueLearner = New(d.Model, d.SessionPoolSize, d.ForceCPU)
		player.ValueScorer = player.ValueLearner
		player.ModelPath = d.Model
	}
}

func ParsePlayerParam(data interface{}, key, value string) {
	d := data.(*ParsingData)
	if key == "tf" {
		d.UseTensorFlow = true
	} else if key == "tf_cpu" {
		d.ForceCPU = true
	} else if key == "model" {
		d.Model = value
	} else if key == "tf_session_pool_size" {
		var err error
		d.SessionPoolSize, err = strconv.Atoi(value)
		if err != nil {
			log.Panicf("Invalid parameter tf_session_pool_size=%s: %v", value, err)
		}
		if d.SessionPoolSize < 1 {
			log.Panicf("Invalid parameter tf_session_pool_size=%s, it must be > 0", value)
		}
	} else {
		log.Panicf("Unknown parameter '%s=%s' passed to tensorflow module.", key, value)
	}
}

func init() {
	for _, key := range []string{"model", "tf", "tf_cpu", "tf_session_pool_size"} {
		players2.RegisterModule(
			"tf", key, NewParsingData, ParsePlayerParam, FinalizeParsing,
			players2.ScorerType)
	}
}
