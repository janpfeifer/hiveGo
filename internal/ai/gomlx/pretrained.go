// Package pretrained embed GoMLX models.
package gomlx

import (
	_ "embed"
)

// Checkpoint for the pre-trained model. These should come from the embedded checkpoint files.
type Checkpoint struct {
	Json   string
	Binary []byte
}

var (
	// PretrainedModels maps model type to a slice of pretrained models.
	PretrainedModels = map[ModelType][]Checkpoint{
		ModelFNN: {
			Checkpoint{fnn0Json, fnn0Bin},
		},
		ModelAlphaZeroFNN: {
			Checkpoint{a0fnn0Json, a0fnn0Bin},
		},
	}
)

var (
	//go:embed pretrained/fnn0.json
	fnn0Json string

	//go:embed pretrained/fnn0.bin
	fnn0Bin []byte

	//go:embed pretrained/a0fnn0.json
	a0fnn0Json string

	//go:embed pretrained/a0fnn0.bin
	a0fnn0Bin []byte
)
