//go:build js && wasm

package gomlx

// SupportedBackends in order of preference for selection.
var SupportedBackends = []string{
	"onnx:wasm",
	"go",
	"onnx:webgpu",
	"onnx:webnn",
}
