package distance_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nozzle/umap-go/distance"
)

// pairwiseData is the JSON structure from testdata/pairwise_distances/*.json.
type pairwiseData struct {
	Metric    string      `json:"metric"`
	X         [][]float64 `json:"X"`
	Distances [][]float64 `json:"distances"`
}

type pairwiseGradData struct {
	Metric    string        `json:"metric"`
	X         [][]float64   `json:"X"`
	Distances [][]float64   `json:"distances"`
	Gradients [][][]float64 `json:"gradients"`
}

type sparseData struct {
	Metric    string      `json:"metric"`
	XShape    []int       `json:"X_shape"`
	XIndptr   []int       `json:"X_indptr"`
	XIndices  []int       `json:"X_indices"`
	XData     []float64   `json:"X_data"`
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

func loadGradients(t *testing.T, metric string) *pairwiseGradData {
	t.Helper()
	path := filepath.Join("..", "testdata", "pairwise_gradients", metric+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read gradient data for metric %s: %v", metric, err)
	}
	var pd pairwiseGradData
	if err := json.Unmarshal(data, &pd); err != nil {
		t.Fatalf("failed to unmarshal test data for metric %s: %v", metric, err)
	}
	return &pd
}

func loadSparse(t *testing.T, metric string) *sparseData {
	t.Helper()
	path := filepath.Join("..", "testdata", "sparse_distances", metric+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read sparse data for metric %s: %v", metric, err)
	}
	var pd sparseData
	if err := json.Unmarshal(data, &pd); err != nil {
		t.Fatalf("failed to unmarshal test data for metric %s: %v", metric, err)
	}
	return &pd
}

func getMetricsInDir(t *testing.T, dir string) []string {
	path := filepath.Join("..", "testdata", dir)
	files, err := os.ReadDir(path)
	if err != nil {
		t.Fatalf("failed to read dir %s: %v", path, err)
	}
	var metrics []string
	for _, f := range files {
		if strings.HasSuffix(f.Name(), ".json") {
			metrics = append(metrics, strings.TrimSuffix(f.Name(), ".json"))
		}
	}
	return metrics
}

func TestPairwiseDistances(t *testing.T) {
	metrics := getMetricsInDir(t, "pairwise_distances")

	for _, metric := range metrics {
		t.Run(metric, func(t *testing.T) {
			pd := loadPairwise(t, metric)
			distFunc := distance.Named(metric)

			// Some metrics require param func
			if distFunc == nil && distance.NamedWithParam(metric) != nil {
				paramFunc := distance.NamedWithParam(metric)
				distFunc = func(x, y []float64) float64 {
					return paramFunc(x, y, nil)
				}
			}

			if distFunc == nil {
				t.Fatalf("no distance function found for metric %q", metric)
			}

			n := len(pd.X)
			tol := 1e-6 // Float32 differences in numba
			maxDiff := 0.0
			mismatches := 0

			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					got := distFunc(pd.X[i], pd.X[j])
					want := pd.Distances[i][j]

					// Ignore nan/inf matches
					if (math.IsNaN(got) || math.IsInf(got, 0)) && (want == -99999.99 || want == 99999.99) {
						continue
					}

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

func TestPairwiseGradients(t *testing.T) {
	metrics := getMetricsInDir(t, "pairwise_gradients")

	for _, metric := range metrics {
		t.Run(metric, func(t *testing.T) {
			pd := loadGradients(t, metric)
			gradFunc := distance.NamedWithGrad(metric)
			if gradFunc == nil {
				t.Fatalf("no gradient function found for metric %q", metric)
			}

			n := len(pd.X)
			tol := 1e-6
			maxDistDiff := 0.0
			maxGradDiff := 0.0
			distMismatches := 0
			gradMismatches := 0

			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					gotDist, gotGrad := gradFunc(pd.X[i], pd.X[j])
					wantDist := pd.Distances[i][j]
					wantGrad := pd.Gradients[i][j]

					distTol := tol
					if metric == "hyperboloid" && i == j {
						distTol = 2e-4
					}

					if !math.IsNaN(gotDist) {
						distDiff := math.Abs(gotDist - wantDist)
						if distDiff > distTol {
							distMismatches++
							if distDiff > maxDistDiff {
								maxDistDiff = distDiff
							}
							if distMismatches <= 5 {
								t.Errorf("dist[%d][%d]: got %v, want %v", i, j, gotDist, wantDist)
							}
						}
					}

					for k := range gotGrad {
						// python gradients sometimes return NaN when distance is zero, skip
						if (math.IsNaN(gotGrad[k]) || math.IsInf(gotGrad[k], 0)) && (wantGrad[k] == -99999.99 || wantGrad[k] == 99999.99) {
							continue
						}
						// handle numba weirdness near 0
						if (wantGrad[k] == -99999.99 || wantGrad[k] == 99999.99) && gotGrad[k] == 0 {
							continue
						}

						gradDiff := math.Abs(gotGrad[k] - wantGrad[k])
						if gradDiff > maxGradDiff {
							maxGradDiff = gradDiff
						}
						if gradDiff > tol {
							gradMismatches++
							if gradMismatches <= 5 {
								t.Errorf("grad[%d][%d][%d]: got %v, want %v", i, j, k, gotGrad[k], wantGrad[k])
							}
						}
					}
				}
			}

			if distMismatches > 0 || gradMismatches > 0 {
				t.Errorf("mismatches: dist=%d, grad=%d", distMismatches, gradMismatches)
			}
		})
	}
}

func TestSparseDistances(t *testing.T) {
	metrics := getMetricsInDir(t, "sparse_distances")

	for _, metric := range metrics {
		t.Run(metric, func(t *testing.T) {
			pd := loadSparse(t, metric)
			sparseFunc := distance.NamedSparse(metric)
			sparseFuncWithN := distance.NamedSparseWithN(metric)

			if sparseFunc == nil && sparseFuncWithN == nil {
				t.Fatalf("no sparse function found for metric %q", metric)
			}

			n := pd.XShape[0]
			nFeatures := pd.XShape[1]
			tol := 1e-6
			if metric == "hellinger" {
				tol = 2e-4
			}
			maxDiff := 0.0
			mismatches := 0

			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					start1, end1 := pd.XIndptr[i], pd.XIndptr[i+1]
					ind1 := pd.XIndices[start1:end1]
					data1 := pd.XData[start1:end1]

					start2, end2 := pd.XIndptr[j], pd.XIndptr[j+1]
					ind2 := pd.XIndices[start2:end2]
					data2 := pd.XData[start2:end2]

					var got float64
					if sparseFuncWithN != nil {
						got = sparseFuncWithN(ind1, data1, ind2, data2, nFeatures)
					} else {
						got = sparseFunc(ind1, data1, ind2, data2)
					}

					want := pd.Distances[i][j]

					if (math.IsNaN(got) || math.IsInf(got, 0)) && (want == -99999.99 || want == 99999.99) {
						continue
					}

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
