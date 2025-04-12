//go:build !nogomlx

package main

// Include GoMLX models support.

import (
	_ "github.com/gomlx/gomlx/backends/xla"
	_ "github.com/janpfeifer/hiveGo/internal/ai/gomlx"
)
