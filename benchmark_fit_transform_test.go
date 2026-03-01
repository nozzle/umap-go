package umap

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"testing"
	"time"

	umaprand "github.com/nozzle/umap-go/rand"
)

type fitTransformBenchCase struct {
	Name        string
	NSamples    int
	NFeatures   int
	NNeighbors  int
	NComponents int
	NEpochs     int
	Seed        uint64
}

type pythonBenchOutput struct {
	Cases []pythonBenchCase `json:"cases"`
}

type pythonBenchCase struct {
	Name        string  `json:"name"`
	MeanNSPerOp float64 `json:"mean_ns_per_op"`
}

func benchmarkCases() []fitTransformBenchCase {
	return []fitTransformBenchCase{
		{
			Name:        "n300_d10_k15_c2_e100",
			NSamples:    300,
			NFeatures:   10,
			NNeighbors:  15,
			NComponents: 2,
			NEpochs:     100,
			Seed:        42,
		},
		{
			Name:        "n600_d20_k15_c2_e100",
			NSamples:    600,
			NFeatures:   20,
			NNeighbors:  15,
			NComponents: 2,
			NEpochs:     100,
			Seed:        42,
		},
	}
}

func benchmarkData(nSamples, nFeatures int, seed uint64) [][]float64 {
	rng := umaprand.NewProduction(&seed)
	data := make([][]float64, nSamples)
	for i := range nSamples {
		data[i] = make([]float64, nFeatures)
		for j := range nFeatures {
			data[i][j] = rng.Float64()
		}
	}
	return data
}

func loadPythonBenchmark() (map[string]float64, error) {
	commands := [][]string{
		{"uv", "run", "python", "testdata/benchmark_fit_transform.py"},
		{"python3", "testdata/benchmark_fit_transform.py"},
	}

	var lastErr error
	for _, args := range commands {
		out, err := exec.Command(args[0], args[1:]...).CombinedOutput()
		if err != nil {
			lastErr = fmt.Errorf("%s failed: %w (%s)", args[0], err, string(out))
			continue
		}

		var parsed pythonBenchOutput
		if err := json.Unmarshal(out, &parsed); err != nil {
			lastErr = fmt.Errorf("%s output parse failed: %w", args[0], err)
			continue
		}

		result := make(map[string]float64, len(parsed.Cases))
		for _, c := range parsed.Cases {
			result[c.Name] = c.MeanNSPerOp
		}
		return result, nil
	}

	return nil, lastErr
}

// BenchmarkFitTransformCompare benchmarks Go FitTransform and reports
// side-by-side Python timing metrics when Python benchmark tooling is available.
func BenchmarkFitTransformCompare(b *testing.B) {
	pythonResults, pyErr := loadPythonBenchmark()
	if pyErr != nil {
		b.Logf("python benchmark unavailable (continuing with Go-only metrics): %v", pyErr)
	}

	for _, tc := range benchmarkCases() {
		b.Run(tc.Name, func(b *testing.B) {
			X := benchmarkData(tc.NSamples, tc.NFeatures, tc.Seed)

			opts := DefaultOptions()
			opts.NNeighbors = tc.NNeighbors
			opts.NComponents = tc.NComponents
			opts.NEpochs = tc.NEpochs
			opts.MinDist = 0.1
			opts.Spread = 1.0

			pyNSPerOp, hasPython := pythonResults[tc.Name]
			if hasPython {
				b.ReportMetric(pyNSPerOp, "py_ns/op")
			}

			b.ReportAllocs()
			b.ResetTimer()
			start := time.Now()
			for range b.N {
				seed := tc.Seed
				opts.RandSource = umaprand.NewProduction(&seed)

				model := New(opts)
				if _, err := model.FitTransform(X, nil); err != nil {
					b.Fatalf("FitTransform failed: %v", err)
				}
			}
			elapsed := time.Since(start)
			b.StopTimer()

			goNSPerOp := float64(elapsed.Nanoseconds()) / float64(b.N)
			b.ReportMetric(goNSPerOp, "go_ns/op")
			if hasPython && goNSPerOp > 0 {
				b.ReportMetric(pyNSPerOp/goNSPerOp, "py/go")
			}
		})
	}
}

// BenchmarkFitTransformParallelModes compares serial vs auto parallel execution
// for the same FitTransform workload.
func BenchmarkFitTransformParallelModes(b *testing.B) {
	tc := benchmarkCases()[0]
	X := benchmarkData(tc.NSamples, tc.NFeatures, tc.Seed)

	for _, mode := range []string{"serial", "auto"} {
		b.Run(mode, func(b *testing.B) {
			opts := DefaultOptions()
			opts.NNeighbors = tc.NNeighbors
			opts.NComponents = tc.NComponents
			opts.NEpochs = tc.NEpochs
			opts.MinDist = 0.1
			opts.Spread = 1.0
			opts.ParallelMode = mode
			if mode == "serial" {
				opts.NWorkers = 1
			}

			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				seed := tc.Seed
				opts.RandSource = umaprand.NewProduction(&seed)
				model := New(opts)
				if _, err := model.FitTransform(X, nil); err != nil {
					b.Fatalf("FitTransform failed: %v", err)
				}
			}
		})
	}
}
