package mcts

import (
	"github.com/janpfeifer/hiveGo/internal/ai"
	"github.com/janpfeifer/hiveGo/internal/parameters"
	"github.com/janpfeifer/hiveGo/internal/searchers"
	"github.com/pkg/errors"
	"time"
)

func NewFromParams(scorer ai.ValueScorer, params parameters.Params) (searchers.Searcher, error) {
	isMCTS, err := parameters.PopParamOr(params, "mcts", false)
	if err != nil {
		return nil, err
	}
	if !isMCTS {
		return nil, nil
	}
	policyScorer, ok := scorer.(ai.PolicyScorer)
	if !ok {
		scale, err := parameters.PopParamOr(params, "policy_scale", float32(1))
		if err != nil {
			return nil, err
		}
		policyScorer = ai.NewPolicyProxy(scorer, scale)
		//return nil, errors.Errorf("mcts requires a 'policy scorer', a normal scorer (%q) won't work", scorer)
	}
	mcts := &mctsSearcher{
		scorer:       policyScorer,
		maxDepth:     10,
		maxTime:      30 * time.Second,
		maxTraverses: 300,
		minTraverses: 0,
		maxAbsScore:  9.0,
		cPuct:        1.1,
		temperature:  0.0,
		parallelized: false,
		useMCTS:      false,
	}
	mcts.cPuct, err = parameters.PopParamOr(params, "c_puct", mcts.cPuct)
	if err != nil {
		return nil, err
	}
	if mcts.cPuct < 0 {
		return nil, errors.Errorf("negative c_puct value (%f given) not possible", mcts.cPuct)
	}
	mcts.maxTime, err = parameters.PopParamOr(params, "max_time", mcts.maxTime)
	if err != nil {
		return nil, err
	}
	mcts.maxDepth, err = parameters.PopParamOr(params, "max_depth", mcts.maxDepth)
	if err != nil {
		return nil, err
	}
	mcts.maxTraverses, err = parameters.PopParamOr(params, "max_traverses", mcts.maxTraverses)
	if err != nil {
		return nil, err
	}
	mcts.temperature, err = parameters.PopParamOr(params, "temperature", mcts.temperature)
	if err != nil {
		return nil, err
	}
	return mcts, nil
}
