package main

import (
	"flag"
	"github.com/janpfeifer/hiveGo/internal/players"
	"github.com/janpfeifer/hiveGo/internal/state"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"strings"
)

var (
	flagAIConfig = flag.String("ai", "", "Configuration for model/searcher to train. "+
		"a0trainer plays against itself, so only one configuration is accepted.")
	flagBootstrapAI = flag.String("bootstrap", "", "Configure an AI to bootstrap an AlphaZero model. "+
		"This is needed because randomly initialized models will never reach a win condition, and never learn anything.")
	// aiPlayer being trained.
	aiPlayer          *players.SearcherScorer
	bootstrapAiPlayer *players.SearcherScorer
)

// Example hold one data point to learn from.
type Example struct {
	board        *state.Board
	valueLabel   float32
	policyLabels []float32
}

func createAIPlayer() error {
	config := *flagAIConfig
	if config == "" {
		return errors.New("must specify AI configuration with -ai")
	}
	if strings.Index(config, ";") != -1 {
		return errors.Errorf("invalid AI config %q, only one AI configuration must be given, no \";\" accepted", config)
	}
	klog.V(1).Infof("Creating AI from %q", config)
	var err error
	aiPlayer, err = players.New(config)
	if err != nil {
		return err
	}
	if aiPlayer.PolicyLearner == nil {
		return errors.Errorf("invalid AI config (-ai): a0trainer requires a \"PolicyLearner\" model, "+
			"but %s doesn't seem to implement it", aiPlayer.ValueScorer)
	}
	if *flagBootstrapAI != "" {
		bootstrapAiPlayer, err = players.New(*flagBootstrapAI)
		if err != nil {
			return errors.WithMessagef(err, "invalid bootstrap AI config (-bootstrap-ai) %q", *flagBootstrapAI)
		}
	}
	return nil
}
