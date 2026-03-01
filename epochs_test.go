package umap

// epochs_test.go tests MakeEpochsPerSample against Python reference data.

import (
	"math"
	"testing"
)

type epochsPerSampleData struct {
	NEpochs         int       `json:"n_epochs"`
	GraphData       []float64 `json:"graph_data"`
	EpochsPerSample []float64 `json:"epochs_per_sample"`
}

func TestMakeEpochsPerSample(t *testing.T) {
	var td epochsPerSampleData
	loadJSON(t, "epochs_per_sample/iris.json", &td)

	got := MakeEpochsPerSample(td.GraphData, td.NEpochs)

	if len(got) != len(td.EpochsPerSample) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(td.EpochsPerSample))
	}

	// Use relative tolerance: upstream graph weights come from fuzzy_simplicial_set
	// which may differ slightly between Go (float64 throughout) and Python
	// (float32 kNN dists promoted to float64). Absolute diffs grow with the
	// magnitude of epochs_per_sample values (which range from 1 to ~900).
	relTol := 1e-5
	for i := range got {
		g, w := got[i], td.EpochsPerSample[i]
		// Both -1 (never sampled)
		if g == -1 && w == -1 {
			continue
		}
		if (g == -1) != (w == -1) {
			t.Errorf("epochs_per_sample[%d]: got %v, want %v (-1 mismatch)", i, g, w)
			continue
		}
		// Relative error
		denom := math.Abs(w)
		if denom < 1e-15 {
			denom = 1e-15
		}
		rel := math.Abs(g-w) / denom
		if rel > relTol {
			t.Errorf("epochs_per_sample[%d]: got %v, want %v (rel=%e, tol=%e)", i, g, w, rel, relTol)
		}
	}
}

func TestMakeEpochsPerSampleProperties(t *testing.T) {
	weights := []float64{1.0, 0.5, 0.25, 0.1, 0.0}
	nEpochs := 200

	result := MakeEpochsPerSample(weights, nEpochs)

	// Maximum weight → epochs_per_sample = 1.0 (sampled every epoch)
	if result[0] != 1.0 {
		t.Errorf("max weight: got %v, want 1.0", result[0])
	}

	// 0 weight → -1 (never sampled)
	if result[4] != -1 {
		t.Errorf("zero weight: got %v, want -1", result[4])
	}

	// Higher weights → lower epochs_per_sample
	for i := 1; i < 4; i++ {
		if result[i] < 0 {
			continue
		}
		if result[i] <= result[i-1] {
			t.Errorf("monotonicity violated: result[%d]=%v <= result[%d]=%v",
				i, result[i], i-1, result[i-1])
		}
	}
}

func TestSelectNEpochs(t *testing.T) {
	tests := []struct {
		n    int
		want int
	}{
		{100, 500},
		{1000, 500},
		{10000, 500},
		{10001, 200},
		{100000, 200},
	}
	for _, tt := range tests {
		got := SelectNEpochs(tt.n)
		if got != tt.want {
			t.Errorf("SelectNEpochs(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

func TestClipValue(t *testing.T) {
	tests := []struct {
		x, limit, want float64
	}{
		{0, 4, 0},
		{5, 4, 4},
		{-5, 4, -4},
		{3.5, 4, 3.5},
		{-3.5, 4, -3.5},
	}
	for _, tt := range tests {
		got := ClipValue(tt.x, tt.limit)
		if got != tt.want {
			t.Errorf("ClipValue(%v, %v) = %v, want %v", tt.x, tt.limit, got, tt.want)
		}
	}
}

func TestRdist(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{4, 5, 6}
	got := Rdist(x, y)
	want := 27.0 // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9+9+9
	if got != want {
		t.Errorf("Rdist = %v, want %v", got, want)
	}
}
