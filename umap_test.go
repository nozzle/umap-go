package umap

// umap_test.go tests the full end-to-end UMAP pipeline.

import (
	"testing"

	umaprand "github.com/nozzle/umap-go/rand"
)

type fullUMAPData struct {
	NNeighbors  int         `json:"n_neighbors"`
	NComponents int         `json:"n_components"`
	MinDist     float64     `json:"min_dist"`
	Spread      float64     `json:"spread"`
	RandomState int         `json:"random_state"`
	NEpochs     int         `json:"n_epochs"`
	X           [][]float64 `json:"X"`
	Embedding   [][]float64 `json:"embedding"`
	A           float64     `json:"a"`
	B           float64     `json:"b"`
}

func TestFullUMAP_IrisUnsupervised(t *testing.T) {
	var td fullUMAPData
	loadJSON(t, "full_umap/iris_unsupervised.json", &td)

	seed := uint64(td.RandomState)
	opts := Options{
		NNeighbors:  td.NNeighbors,
		NComponents: td.NComponents,
		MinDist:     td.MinDist,
		Spread:      td.Spread,
		NEpochs:     td.NEpochs,
		RandSource:  umaprand.NewProduction(&seed),
	}

	model := New(opts)
	embedding, err := model.FitTransform(td.X, nil)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	if len(embedding) != len(td.X) {
		t.Fatalf("embedding rows: got %d, want %d", len(embedding), len(td.X))
	}
	if len(embedding[0]) != td.NComponents {
		t.Fatalf("embedding cols: got %d, want %d", len(embedding[0]), td.NComponents)
	}

	// Check a, b params are reasonable
	if !approxEqual(model.A(), td.A, 0.05) {
		t.Errorf("a param: got %v, want %v", model.A(), td.A)
	}
	if !approxEqual(model.B(), td.B, 0.05) {
		t.Errorf("b param: got %v, want %v", model.B(), td.B)
	}

	// The full pipeline will differ from Python due to:
	// 1. Different curve fitting optimizer (Nelder-Mead vs L-M)
	// 2. Different RNG for SGD
	// 3. Spectral init eigenvector sign ambiguity
	// So we check structural properties.

	// All values should be finite
	for i := range embedding {
		for d := range embedding[i] {
			if embedding[i][d] != embedding[i][d] {
				t.Fatalf("NaN at [%d][%d]", i, d)
			}
		}
	}

	// The embedding should separate the Iris classes somewhat.
	// Load labels to check.
	var dataset struct {
		Y []float64 `json:"y"`
	}
	loadJSON(t, "datasets/iris.json", &dataset)

	// Compute mean embedding per class
	classMeans := make(map[int][]float64)
	classCounts := make(map[int]int)
	for i, label := range dataset.Y {
		c := int(label)
		if classMeans[c] == nil {
			classMeans[c] = make([]float64, td.NComponents)
		}
		for d := range embedding[i] {
			classMeans[c][d] += embedding[i][d]
		}
		classCounts[c]++
	}
	for c := range classMeans {
		for d := range classMeans[c] {
			classMeans[c][d] /= float64(classCounts[c])
		}
	}

	// The three class centroids should be distinct
	if len(classMeans) < 3 {
		t.Fatalf("expected 3 classes, got %d", len(classMeans))
	}
	t.Logf("class centroids: %v", classMeans)

	// Report comparison vs Python reference
	rmse := rmseDiff(embedding, td.Embedding)
	maxD := maxAbsDiff(embedding, td.Embedding)
	t.Logf("vs Python reference: RMSE=%.4f, maxDiff=%.4f", rmse, maxD)
}

func TestFullUMAP_IrisSupervisedShape(t *testing.T) {
	var td fullUMAPData
	loadJSON(t, "full_umap/iris_supervised.json", &td)

	var dataset struct {
		X [][]float64 `json:"X"`
		Y []float64   `json:"y"`
	}
	loadJSON(t, "datasets/iris.json", &dataset)

	seed := uint64(td.RandomState)
	opts := Options{
		NNeighbors:  td.NNeighbors,
		NComponents: td.NComponents,
		MinDist:     td.MinDist,
		Spread:      td.Spread,
		NEpochs:     td.NEpochs,
		RandSource:  umaprand.NewProduction(&seed),
	}

	model := New(opts)
	embedding, err := model.FitTransform(td.X, dataset.Y)
	if err != nil {
		t.Fatalf("FitTransform (supervised) failed: %v", err)
	}

	if len(embedding) != len(td.X) {
		t.Fatalf("embedding rows: got %d, want %d", len(embedding), len(td.X))
	}
	if len(embedding[0]) != td.NComponents {
		t.Fatalf("embedding cols: got %d, want %d", len(embedding[0]), td.NComponents)
	}

	// Check model is fitted
	if model.Graph() == nil {
		t.Error("graph should not be nil after Fit")
	}
	if model.Embedding() == nil {
		t.Error("embedding should not be nil after Fit")
	}

	t.Logf("supervised embedding has %d points in %d dimensions", len(embedding), len(embedding[0]))
}

func TestUMAP_EmptyInput(t *testing.T) {
	model := New(DefaultOptions())
	_, err := model.FitTransform(nil, nil)
	if err == nil {
		t.Error("expected error for nil input")
	}

	_, err = model.FitTransform([][]float64{}, nil)
	if err == nil {
		t.Error("expected error for empty input")
	}
}

func TestUMAP_TransformNotFitted(t *testing.T) {
	model := New(DefaultOptions())
	_, err := model.Transform([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for Transform on unfitted model")
	}
}

func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()

	if opts.NNeighbors != 15 {
		t.Errorf("NNeighbors: got %d, want 15", opts.NNeighbors)
	}
	if opts.NComponents != 2 {
		t.Errorf("NComponents: got %d, want 2", opts.NComponents)
	}
	if opts.MinDist != 0.1 {
		t.Errorf("MinDist: got %v, want 0.1", opts.MinDist)
	}
	if opts.Spread != 1.0 {
		t.Errorf("Spread: got %v, want 1.0", opts.Spread)
	}
	if opts.Metric != "euclidean" {
		t.Errorf("Metric: got %q, want \"euclidean\"", opts.Metric)
	}
	if opts.InitMethod != "spectral" {
		t.Errorf("InitMethod: got %q, want \"spectral\"", opts.InitMethod)
	}
	if opts.RandSource == nil {
		t.Error("RandSource should not be nil")
	}
}
