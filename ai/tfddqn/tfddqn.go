// tfddqn Learns with Double Q-Learning.
// See paper in https://arxiv.org/abs/1509.06461
package tfddqn

import "github.com/janpfeifer/hiveGo/ai/tensorflow"

type Scorer struct{
	models [2]*tensorflow.Scorer
}

func New(basename string, session) (scorer *Scorer) {
	scorer = &Scorer{}
	names := []string{
		basename + "_a",
		basename + "_b",
	}
	for ii, name := range names {
		scorer.models[ii] = tensorflow.New(name)
	}
}