// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !js && !wasm

package bench

// benchmarkBackends for WASM
var benchmarkBackends = []string{
	"go",
	"xla:cpu",
	"xla:cuda",
	"onnx:cpu",
	"onnx:cuda",
}
