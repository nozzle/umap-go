package distance_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/nozzle/umap-go/distance"
)

// pairwiseData is the JSON structure from testdata/pairwise_distances/*.json.
type pairwiseData struct {
	Metric    string      `json:"metric"`
	X         [][]float64 `json:"X"`
	Distances [][]float64 `json:"distances"`
}

func loadPairwise(t *testing.T, metric string) *pairwiseData {
	t.Helper()
	path := filepath.Join("..", "testdata", "pairwise_distances", metric+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read test data for metric %s: %v", metric, err)
	}
	var pd pairwiseData
	if err := json.Unmarshal(data, &pd); err != nil {
		t.Fatalf("failed to unmarshal test data for metric %s: %v", metric, err)
	}
	return &pd
}

func TestPairwiseDistances(t *testing.T) {
	metrics := []string{
		"euclidean", "manhattan", "cosine", "chebyshev",
		"canberra", "braycurtis", "hamming", "jaccard",
	}

	for _, metric := range metrics {
		t.Run(metric, func(t *testing.T) {
			pd := loadPairwise(t, metric)
			distFunc := distance.Named(metric)
			if distFunc == nil {
				t.Fatalf("no distance function found for metric %q", metric)
			}

			n := len(pd.X)
			tol := 1e-10
			maxDiff := 0.0
			mismatches := 0

			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					got := distFunc(pd.X[i], pd.X[j])
					want := pd.Distances[i][j]
					diff := math.Abs(got - want)
					if diff > maxDiff {
						maxDiff = diff
					}
					if diff > tol {
						mismatches++
						if mismatches <= 5 {
							t.Errorf("[%d][%d]: got %v, want %v (diff=%v)", i, j, got, want, diff)
						}
					}
				}
			}

			if mismatches > 5 {
				t.Errorf("... and %d more mismatches (max diff=%v)", mismatches-5, maxDiff)
			}
			if mismatches == 0 {
				t.Logf("all %dx%d distances match within tol=%v (max diff=%v)", n, n, tol, maxDiff)
			}
		})
	}
}

func TestDistanceRegistryCompleteness(t *testing.T) {
	// Verify all expected metrics are registered
	expected := []string{
		"euclidean", "l2", "manhattan", "taxicab", "l1",
		"chebyshev", "linfinity", "linfty", "linf",
		"canberra", "braycurtis", "cosine", "correlation",
		"hellinger", "haversine", "poincare",
		"jaccard", "dice", "hamming", "matching",
		"kulsinski", "rogerstanimoto", "russellrao",
		"sokalsneath", "sokalmichener", "yule",
	}

	for _, name := range expected {
		if !distance.IsValid(name) {
			t.Errorf("metric %q not registered as valid", name)
		}
		fn := distance.Named(name)
		if fn == nil {
			t.Errorf("metric %q returned nil from Named()", name)
		}
	}
}

func TestGradientRegistryCompleteness(t *testing.T) {
	expectedGrad := []string{
		"euclidean", "l2", "manhattan", "taxicab", "l1",
		"chebyshev", "linfinity", "linfty", "linf",
		"cosine", "correlation", "canberra", "braycurtis",
		"hellinger", "haversine", "hyperboloid",
	}

	for _, name := range expectedGrad {
		if !distance.HasGradient(name) {
			t.Errorf("metric %q should have gradient but HasGradient returned false", name)
		}
		fn := distance.NamedWithGrad(name)
		if fn == nil {
			t.Errorf("metric %q returned nil from NamedWithGrad()", name)
		}
	}
}

func TestSparseRegistryCompleteness(t *testing.T) {
	expectedSparse := []string{
		"euclidean", "l2", "manhattan", "taxicab", "l1",
		"chebyshev", "linfinity", "linfty", "linf",
		"canberra", "braycurtis", "cosine",
		"hellinger", "jaccard", "dice", "sokalsneath",
		"ll_dirichlet",
	}

	for _, name := range expectedSparse {
		if !distance.HasSparse(name) {
			t.Errorf("metric %q should have sparse variant but HasSparse returned false", name)
		}
	}

	expectedSparseWithN := []string{
		"hamming", "matching", "kulsinski", "rogerstanimoto",
		"russellrao", "sokalmichener", "correlation",
	}

	for _, name := range expectedSparseWithN {
		if !distance.SparseNeedsNFeatures(name) {
			t.Errorf("metric %q should need n_features but SparseNeedsNFeatures returned false", name)
		}
	}
}

func TestEuclideanBasic(t *testing.T) {
	x := []float64{0, 0}
	y := []float64{3, 4}
	got := distance.Euclidean(x, y)
	want := 5.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Euclidean([0,0],[3,4]) = %v, want %v", got, want)
	}
}

func TestCosineBasic(t *testing.T) {
	x := []float64{1, 0}
	y := []float64{0, 1}
	got := distance.Cosine(x, y)
	want := 1.0 // orthogonal vectors
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Cosine([1,0],[0,1]) = %v, want %v", got, want)
	}

	// Same direction
	got = distance.Cosine(x, x)
	want = 0.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Cosine([1,0],[1,0]) = %v, want %v", got, want)
	}
}

func TestGradientConsistency(t *testing.T) {
	// Test that gradient functions return the same distance as non-gradient versions
	x := []float64{1.5, 2.3, -0.7, 4.1}
	y := []float64{-0.5, 3.1, 0.2, 2.8}

	tests := []struct {
		name string
		fn   distance.Func
		grad distance.GradFunc
	}{
		{"euclidean", distance.Euclidean, distance.EuclideanGrad},
		{"manhattan", distance.Manhattan, distance.ManhattanGrad},
		{"chebyshev", distance.Chebyshev, distance.ChebyshevGrad},
		{"cosine", distance.Cosine, distance.CosineGrad},
		{"canberra", distance.Canberra, distance.CanberraGrad},
		{"braycurtis", distance.BrayCurtis, distance.BrayCurtisGrad},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			want := tt.fn(x, y)
			got, grad := tt.grad(x, y)
			if math.Abs(got-want) > 1e-10 {
				t.Errorf("distance mismatch: grad returned %v, fn returned %v", got, want)
			}
			if len(grad) != len(x) {
				t.Errorf("gradient length: got %d, want %d", len(grad), len(x))
			}
		})
	}
}
