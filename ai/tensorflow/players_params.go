package tensorflow

import (
	"github.com/janpfeifer/hiveGo/ai/players"
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

func FinalizeParsing(data interface{}, player *players.SearcherScorer) {
	d := data.(*ParsingData)
	if d.UseTensorFlow {
		player.Learner = New(d.Model, d.SessionPoolSize, d.ForceCPU)
		player.Scorer = player.Learner
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
		players.RegisterPlayerModule(
			"tf", key, NewParsingData, ParsePlayerParam, FinalizeParsing,
			players.ScorerType)
	}
}
