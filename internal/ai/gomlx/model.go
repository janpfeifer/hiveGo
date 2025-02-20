package gomlx

import "github.com/gomlx/gomlx/ml/context"

// Model is a GoMLX supported model, the backend of the gomlx.Scorer.
type Model interface {
	// CreateContext with default parameters.
	CreateContext() *context.Context
}
