//go:build !js || !wasm

package gomlx

// SupportedBackends in order of preference for selection.
var SupportedBackends = []string{
	"xla:cuda",
	"xla:cpu",
	"onnx:cpu",
	"onnx:cuda",
	"go",
}
