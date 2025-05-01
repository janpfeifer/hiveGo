//go:build !nogomlx

package main

// Include GoMLX backend and models support.

import (
	_ "github.com/gomlx/gomlx/backends/simplego"
	_ "github.com/janpfeifer/hiveGo/internal/ai/gomlx"
)
