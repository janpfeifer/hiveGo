# HiveGo GoMLX Model Inference Benchmarks

This package benchmarks pure model inference performance across different computation backends for HiveGo's pre-trained neural network models:
- **`fnn0` (FNN)**: Feed-forward model for Alpha-Beta search evaluation (`FNN0_ValueScore`).
- **`a0fnn0` (AlphaZeroFNN)**: Dual-head model used for MCTS search:
  - `A0FNN0_ValueScore`: Board state position evaluation.
  - `A0FNN0_PolicyScore`: Message-passing policy distribution over legal actions.

The benchmark isolates pure model execution time (`Exec.Call`) by replaying pre-extracted feature tensors recorded during a simulated match, avoiding game search tree traversal overhead.

---

## 1. Running Native Benchmarks (Desktop / Server)

Runs natively on CPU (and CUDA GPU if available):

```bash
# Run all native backends (e.g. "go", "xla", "onnx:cpu", "onnx:cuda")
go test -bench=. ./internal/ai/gomlx/bench

# Run a specific model or backend
go test -bench='BenchmarkInference/Backend=onnx' ./internal/ai/gomlx/bench
```

---

## 2. Running WebAssembly (WASM / Browser) Benchmarks

WebAssembly tests run inside a real browser engine using [`wasmbrowsertest`](https://github.com/agnivade/wasmbrowsertest).

### Installation of `wasmbrowsertest`
For compatibility with modern Chromium (v130+), build with the latest protocol definitions:

```bash
mkdir -p /tmp/wbt && cd /tmp/wbt
go mod init wbt
go get github.com/agnivade/wasmbrowsertest@master \
       github.com/chromedp/cdproto@latest \
       github.com/chromedp/chromedp@latest
go install github.com/agnivade/wasmbrowsertest
cd - && rm -rf /tmp/wbt
```

---

### Option A: Headless Mode (Default CPU)

Runs headless Chrome in the background. Tests the pure Go backend and the ONNX Runtime Web WASM SIMD execution provider:

```bash
GOOS=js GOARCH=wasm go test -bench=. -exec wasmbrowsertest ./internal/ai/gomlx/bench
```

*Note: Headless Chrome disables GPU hardware acceleration by default, so `onnx:webgpu` and `onnx:webnn` are automatically skipped in headless mode.*

---

### Option B: GUI Mode with WebGPU (`WASM_HEADLESS=off`)

Setting `WASM_HEADLESS=off` launches a foreground Chrome browser window with full hardware GPU acceleration enabled:

```bash
WASM_HEADLESS=off GOOS=js GOARCH=wasm go test -bench=. -exec wasmbrowsertest ./internal/ai/gomlx/bench
```

To run only the WebGPU backend:
```bash
WASM_HEADLESS=off GOOS=js GOARCH=wasm go test -bench='Backend=onnx:webgpu' -exec wasmbrowsertest ./internal/ai/gomlx/bench
```

---

## 3. Re-extracting Feature Datasets

To record a new sequence of feature tensors from a simulated match between two AI players:

```bash
go run ./cmd/hive --watch \
  --config="fnn=#0,ab,max_depth=2" \
  --config2="a0fnn=#0,mcts,max_time=200ms" \
  --max_moves=30 \
  --quiet \
  --save_features=./internal/ai/gomlx/bench/match_features.bin
```
