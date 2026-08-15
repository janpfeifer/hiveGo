//go:build !nogomlx

package main

// Include GoMLX backends and models support.

import (
	_ "github.com/gomlx/gomlx/backends/default"
	_ "github.com/janpfeifer/hiveGo/internal/ai/gomlx"
)
