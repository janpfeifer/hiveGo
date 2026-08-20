package bench

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute-onnx"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/janpfeifer/hiveGo/internal/ai/gomlx"
	"github.com/janpfeifer/hiveGo/internal/parameters"
)

var (
	testDataset *MatchDataset
)

func init() {
	var err error
	testDataset, err = LoadEmbedded()
	if err != nil {
		panic(fmt.Sprintf("Failed to load embedded benchmark dataset: %+v", err))
	}
}

func getBackend(backendName string) (compute.Backend, error) {
	return compute.NewWithConfig(backendName)
}

func BenchmarkInference(b *testing.B) {
	for _, backendName := range benchmarkBackends {
		b.Run(fmt.Sprintf("Backend=%s", backendName), func(b *testing.B) {
			backend, err := getBackend(backendName)
			if err != nil {
				b.Skipf("Backend %q not available: %v", backendName, err)
				return
			}
			defer backend.Finalize()

			// Pre-convert all dataset tensors
			fnnInputsList := make([][]*tensors.Tensor, len(testDataset.Steps))
			fnnBoardsPerStep := make([]int, len(testDataset.Steps))
			a0ValInputsList := make([][]*tensors.Tensor, len(testDataset.Steps))
			a0ValBoardsPerStep := make([]int, len(testDataset.Steps))
			a0PolInputsList := make([][]*tensors.Tensor, len(testDataset.Steps))
			a0PolActionsPerStep := make([]int, len(testDataset.Steps))

			for i, step := range testDataset.Steps {
				for _, rt := range step.FNNInputs {
					fnnInputsList[i] = append(fnnInputsList[i], rt.ToTensor())
				}
				if len(step.FNNInputs) > 1 && len(step.FNNInputs[1].Int32) > 0 {
					fnnBoardsPerStep[i] = int(step.FNNInputs[1].Int32[0])
				}

				for _, rt := range step.A0ValInputs {
					a0ValInputsList[i] = append(a0ValInputsList[i], rt.ToTensor())
				}
				if len(step.A0ValInputs) > 1 && len(step.A0ValInputs[1].Int32) > 0 {
					a0ValBoardsPerStep[i] = int(step.A0ValInputs[1].Int32[0])
				}

				for _, rt := range step.A0PolInputs {
					a0PolInputsList[i] = append(a0PolInputsList[i], rt.ToTensor())
				}
				if len(step.A0PolInputs) > 4 && len(step.A0PolInputs[4].Int32) > 0 {
					a0PolActionsPerStep[i] = int(step.A0PolInputs[4].Int32[0])
				}
			}

			// 1. Benchmark FNN0 Board Scorer
			b.Run("FNN0_ValueScore", func(b *testing.B) {
				fnnScorer, err := gomlx.NewBoardScorerWithBackend(gomlx.ModelFNN, "#0", gomlx.NewFNN(), parameters.Params{}, backend)
				if err != nil {
					b.Fatalf("Failed to create FNN0 scorer: %+v", err)
				}
				scoreExec := fnnScorer.ScoreExec()

				// Warmup and pre-compile across all distinct input sizes
				for _, inputs := range fnnInputsList {
					inps := make([]any, len(inputs))
					for j, t := range inputs {
						buf, _ := graph.DonateTensorBuffer(t, backend, 0)
						inps[j] = buf
					}
					_ = scoreExec.MustCall(inps...)
				}

				var totalBoards int64
				b.ResetTimer()
				stepIdx := 0
				for i := 0; i < b.N; i++ {
					idx := stepIdx % len(fnnInputsList)
					inputs := fnnInputsList[idx]
					inps := make([]any, len(inputs))
					for j, t := range inputs {
						buf, _ := graph.DonateTensorBuffer(t, backend, 0)
						inps[j] = buf
					}
					_ = scoreExec.MustCall(inps...)
					totalBoards += int64(fnnBoardsPerStep[idx])
					stepIdx++
				}
				b.StopTimer()

				if totalBoards > 0 && b.Elapsed() > 0 {
					nsPerBoard := float64(b.Elapsed().Nanoseconds()) / float64(totalBoards)
					b.ReportMetric(nsPerBoard, "ns/board")
					b.ReportMetric(float64(totalBoards)/(b.Elapsed().Seconds()*1000), "kboards/s")
				}
			})

			// 2. Benchmark A0FNN0 Value Score
			b.Run("A0FNN0_ValueScore", func(b *testing.B) {
				a0Scorer, err := gomlx.NewPolicyScorerWithBackend(gomlx.ModelAlphaZeroFNN, "#0", gomlx.NewAlphaZeroFNN(), parameters.Params{}, backend)
				if err != nil {
					b.Fatalf("Failed to create A0FNN0 scorer: %+v", err)
				}
				valExec := a0Scorer.ValueScoreExec()

				// Warmup and pre-compile
				for _, inputs := range a0ValInputsList {
					inps := make([]any, len(inputs))
					for j, t := range inputs {
						buf, _ := graph.DonateTensorBuffer(t, backend, 0)
						inps[j] = buf
					}
					_ = valExec.MustCall(inps...)
				}

				var totalBoards int64
				b.ResetTimer()
				stepIdx := 0
				for i := 0; i < b.N; i++ {
					idx := stepIdx % len(a0ValInputsList)
					inputs := a0ValInputsList[idx]
					inps := make([]any, len(inputs))
					for j, t := range inputs {
						buf, _ := graph.DonateTensorBuffer(t, backend, 0)
						inps[j] = buf
					}
					_ = valExec.MustCall(inps...)
					totalBoards += int64(a0ValBoardsPerStep[idx])
					stepIdx++
				}
				b.StopTimer()

				if totalBoards > 0 && b.Elapsed() > 0 {
					nsPerBoard := float64(b.Elapsed().Nanoseconds()) / float64(totalBoards)
					b.ReportMetric(nsPerBoard, "ns/board")
					b.ReportMetric(float64(totalBoards)/(b.Elapsed().Seconds()*1000), "kboards/s")
				}
			})

			// 3. Benchmark A0FNN0 Policy Score
			b.Run("A0FNN0_PolicyScore", func(b *testing.B) {
				a0Scorer, err := gomlx.NewPolicyScorerWithBackend(gomlx.ModelAlphaZeroFNN, "#0", gomlx.NewAlphaZeroFNN(), parameters.Params{}, backend)
				if err != nil {
					b.Fatalf("Failed to create A0FNN0 scorer: %+v", err)
				}
				polExec := a0Scorer.PolicyScoreExec()

				// Warmup and pre-compile across all padded action sizes
				for _, inputs := range a0PolInputsList {
					inps := make([]any, len(inputs))
					for j, t := range inputs {
						buf, _ := graph.DonateTensorBuffer(t, backend, 0)
						inps[j] = buf
					}
					_ = polExec.MustCall(inps...)
				}

				var totalActions int64
				b.ResetTimer()
				stepIdx := 0
				for i := 0; i < b.N; i++ {
					idx := stepIdx % len(a0PolInputsList)
					inputs := a0PolInputsList[idx]
					inps := make([]any, len(inputs))
					for j, t := range inputs {
						buf, _ := graph.DonateTensorBuffer(t, backend, 0)
						inps[j] = buf
					}
					_ = polExec.MustCall(inps...)
					totalActions += int64(a0PolActionsPerStep[idx])
					stepIdx++
				}
				b.StopTimer()

				if totalActions > 0 && b.Elapsed() > 0 {
					nsPerAction := float64(b.Elapsed().Nanoseconds()) / float64(totalActions)
					b.ReportMetric(nsPerAction, "ns/action")
					b.ReportMetric(float64(totalActions)/(b.Elapsed().Seconds()*1000), "kactions/s")
				}
			})
		})
	}
}
