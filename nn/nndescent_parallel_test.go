package nn_test

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/nozzle/umap-go/distance"
	"github.com/nozzle/umap-go/nn"
)

func TestNNDescentParallelMatchesSerial(t *testing.T) {
	data := buildParallelTestData(48, 5)

	serialCfg := nn.NNDescentConfig{
		K:             6,
		MaxCandidates: 6,
		NIters:        3,
		NWorkers:      1,
		ParallelMode:  "serial",
	}
	parallelCfg := serialCfg
	parallelCfg.NWorkers = 4
	parallelCfg.ParallelMode = "parallel"

	serial := nn.NNDescent(data, distance.Euclidean, serialCfg)
	parallel := nn.NNDescent(data, distance.Euclidean, parallelCfg)

	for i := range len(serial.Indices) {
		for j := range len(serial.Indices[i]) {
			if serial.Indices[i][j] != parallel.Indices[i][j] {
				t.Fatalf("indices mismatch at [%d][%d]: serial=%d parallel=%d", i, j, serial.Indices[i][j], parallel.Indices[i][j])
			}
			if serial.Distances[i][j] != parallel.Distances[i][j] {
				t.Fatalf("distances mismatch at [%d][%d]: serial=%v parallel=%v", i, j, serial.Distances[i][j], parallel.Distances[i][j])
			}
		}
	}
}

func TestNNDescentParallelRunsWorkers(t *testing.T) {
	data := buildParallelTestData(48, 5)

	var active int64
	var maxActive int64
	parallelDist := func(x, y []float64) float64 {
		cur := atomic.AddInt64(&active, 1)
		for {
			prev := atomic.LoadInt64(&maxActive)
			if cur <= prev || atomic.CompareAndSwapInt64(&maxActive, prev, cur) {
				break
			}
		}
		time.Sleep(200 * time.Microsecond)
		d := distance.Euclidean(x, y)
		atomic.AddInt64(&active, -1)
		return d
	}

	cfg := nn.NNDescentConfig{
		K:             6,
		MaxCandidates: 6,
		NIters:        2,
		NWorkers:      4,
		ParallelMode:  "parallel",
	}
	_ = nn.NNDescent(data, parallelDist, cfg)

	if got := atomic.LoadInt64(&maxActive); got < 2 {
		t.Fatalf("expected parallel distance evaluation, max concurrent calls=%d", got)
	}
}

func buildParallelTestData(n, dim int) [][]float64 {
	data := make([][]float64, n)
	for i := range n {
		row := make([]float64, dim)
		for d := range dim {
			row[d] = float64((i+1)*(d+3)%17) + float64(i)/10.0 + float64(d)/100.0
		}
		data[i] = row
	}
	return data
}
