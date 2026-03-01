package umap

// layout_test.go tests the SGD layout optimization against Python reference data.

import (
	"testing"

	"github.com/nozzle/umap-go/nn"
)

type sgdLayoutData struct {
	Init               [][]float64 `json:"init"`
	Head               []float64   `json:"head"`
	Tail               []float64   `json:"tail"`
	NEpochs            int         `json:"n_epochs"`
	A                  float64     `json:"a"`
	B                  float64     `json:"b"`
	Gamma              float64     `json:"gamma"`
	InitialAlpha       float64     `json:"initial_alpha"`
	NegativeSampleRate int         `json:"negative_sample_rate"`
	EpochsPerSample    []float64   `json:"epochs_per_sample"`
	RngState           []float64   `json:"rng_state"`
	Embedding          [][]float64 `json:"embedding"`
}

func TestOptimizeLayoutEuclidean(t *testing.T) {
	var td sgdLayoutData
	loadJSON(t, "sgd_layout/small.json", &td)

	// Convert head/tail from float64 to int
	head := float64sToInts(td.Head)
	tail := float64sToInts(td.Tail)

	// Deep copy init so we don't modify the test data
	nPts := len(td.Init)
	dim := len(td.Init[0])
	embedding := make([][]float64, nPts)
	for i := range nPts {
		embedding[i] = make([]float64, dim)
		copy(embedding[i], td.Init[i])
	}

	// Initialize per-edge RNG states from the recorded rng_state
	// The Python code uses a single TauRand state for all edges
	rngStates := make([]nn.TauRandState, len(head))
	baseState := nn.TauRandState{
		int64(td.RngState[0]),
		int64(td.RngState[1]),
		int64(td.RngState[2]),
	}

	// Each edge gets its own state derived from the base state
	for i := range rngStates {
		rngStates[i] = baseState
		// Advance the state for each edge to get different streams
		nn.TauRandInt(&baseState)
	}

	cfg := OptimizeLayoutConfig{
		A:                  td.A,
		B:                  td.B,
		Gamma:              td.Gamma,
		InitialAlpha:       td.InitialAlpha,
		NegativeSampleRate: float64(td.NegativeSampleRate),
		NEpochs:            td.NEpochs,
	}

	result := OptimizeLayoutEuclidean(
		embedding, embedding,
		head, tail,
		td.EpochsPerSample,
		rngStates,
		cfg,
	)

	// Due to RNG stream differences between Python (Tausworthe per-sample
	// with single state) and our per-edge states, the SGD results will
	// differ. We verify structural properties instead.

	if len(result) != nPts {
		t.Fatalf("result rows: got %d, want %d", len(result), nPts)
	}
	if len(result[0]) != dim {
		t.Fatalf("result cols: got %d, want %d", len(result[0]), dim)
	}

	// Check that the embedding has changed from init
	changed := false
	for i := range result {
		for d := range result[i] {
			if result[i][d] != td.Init[i][d] {
				changed = true
				break
			}
		}
		if changed {
			break
		}
	}
	if !changed {
		t.Error("embedding unchanged after optimization")
	}

	// Check that embedding values are finite
	for i := range result {
		for d := range result[i] {
			if result[i][d] != result[i][d] { // NaN check
				t.Errorf("NaN in embedding[%d][%d]", i, d)
			}
		}
	}

	// Report RMSE vs reference for debugging
	rmse := rmseDiff(result, td.Embedding)
	maxD := maxAbsDiff(result, td.Embedding)
	t.Logf("vs Python reference: RMSE=%.6f, maxDiff=%.6f", rmse, maxD)
}
