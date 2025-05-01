//go:build !nogomlx && xla

package main

// Include GoMLX backend and models support.

import (
	_ "github.com/gomlx/gomlx/backends/xla"
)
