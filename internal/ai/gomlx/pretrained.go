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
	}
)

//go:embed pretrained/fnn0.json
var fnn0Json string

//go:embed pretrained/fnn0.bin
var fnn0Bin []byte
