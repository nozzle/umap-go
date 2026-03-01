package nn_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/nozzle/umap-go/distance"
	"github.com/nozzle/umap-go/nn"
)

type knnBruteData struct {
	X          [][]float64 `json:"X"`
	NNeighbors int         `json:"n_neighbors"`
	Metric     string      `json:"metric"`
	KNNIndices [][]float64 `json:"knn_indices"`
	KNNDists   [][]float64 `json:"knn_dists"`
}

func loadKNNData(t *testing.T, dir, file string) *knnBruteData {
	t.Helper()
	path := filepath.Join("..", "testdata", dir, file)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read test data: %v", err)
	}
	var kd knnBruteData
	if err := json.Unmarshal(data, &kd); err != nil {
		t.Fatalf("failed to unmarshal test data: %v", err)
	}
	return &kd
}

func loadKNNBrute(t *testing.T) *knnBruteData {
	t.Helper()
	path := filepath.Join("..", "testdata", "knn_brute", "euclidean.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read test data: %v", err)
	}
	var kd knnBruteData
	if err := json.Unmarshal(data, &kd); err != nil {
		t.Fatalf("failed to unmarshal test data: %v", err)
	}
	return &kd
}

func TestBruteForceKNN(t *testing.T) {
	kd := loadKNNBrute(t)

	indices, distances := nn.BruteForceKNN(kd.X, kd.NNeighbors, distance.Euclidean)

	n := len(kd.X)
	k := kd.NNeighbors

	if len(indices) != n || len(distances) != n {
		t.Fatalf("result shape: got %dx?, want %dx%d", len(indices), n, k)
	}

	// Compare indices
	indicesMismatch := 0
	for i := range n {
		if len(indices[i]) != k {
			t.Fatalf("indices[%d] length: got %d, want %d", i, len(indices[i]), k)
		}
		for j := range k {
			wantIdx := int(kd.KNNIndices[i][j])
			if indices[i][j] != wantIdx {
				indicesMismatch++
				if indicesMismatch <= 5 {
					t.Errorf("indices[%d][%d]: got %d, want %d", i, j, indices[i][j], wantIdx)
				}
			}
		}
	}
	if indicesMismatch > 5 {
		t.Errorf("... and %d more index mismatches", indicesMismatch-5)
	}

	// Compare distances
	distMismatch := 0
	tol := 1e-6 // Python kNN distances are float32; Go uses float64
	for i := range n {
		for j := range k {
			diff := math.Abs(distances[i][j] - kd.KNNDists[i][j])
			if diff > tol {
				distMismatch++
				if distMismatch <= 5 {
					t.Errorf("distances[%d][%d]: got %v, want %v (diff=%v)",
						i, j, distances[i][j], kd.KNNDists[i][j], diff)
				}
			}
		}
	}
	if distMismatch > 5 {
		t.Errorf("... and %d more distance mismatches", distMismatch-5)
	}

	if indicesMismatch == 0 && distMismatch == 0 {
		t.Logf("all %dx%d kNN results match exactly", n, k)
	}
}

func TestPairwiseDistances(t *testing.T) {
	kd := loadKNNBrute(t)

	D := nn.PairwiseDistances(kd.X, distance.Euclidean)
	n := len(kd.X)

	// D should be n x n
	if len(D) != n {
		t.Fatalf("D rows: got %d, want %d", len(D), n)
	}

	// Diagonal should be zero
	for i := range n {
		if D[i][i] != 0 {
			t.Errorf("D[%d][%d] = %v, want 0", i, i, D[i][i])
		}
	}

	// Should be symmetric
	for i := range n {
		for j := i + 1; j < n; j++ {
			if math.Abs(D[i][j]-D[j][i]) > 1e-12 {
				t.Errorf("asymmetric: D[%d][%d]=%v != D[%d][%d]=%v", i, j, D[i][j], j, i, D[j][i])
			}
		}
	}
}

func TestTauRand(t *testing.T) {
	// Test that TauRandInt produces deterministic output from a given state
	state := nn.TauRandState{42, 13, 7}

	// Generate a sequence
	vals := make([]int64, 100)
	for i := range vals {
		vals[i] = nn.TauRandInt(&state)
	}

	// Re-seed with same state
	state2 := nn.TauRandState{42, 13, 7}
	for i := range vals {
		v := nn.TauRandInt(&state2)
		if v != vals[i] {
			t.Fatalf("non-deterministic at %d: got %d, want %d", i, v, vals[i])
		}
	}

	// TauRandIntRange should produce values in [0, n)
	state3 := nn.TauRandState{42, 13, 7}
	n := 100
	for range 1000 {
		v := nn.TauRandIntRange(&state3, n)
		if v < 0 || v >= n {
			t.Fatalf("TauRandIntRange out of range: got %d, want [0,%d)", v, n)
		}
	}
}

func TestNNDescentKNN(t *testing.T) {
	kd := loadKNNData(t, "knn_nndescent", "euclidean.json")

	pythonVals := []int64{
		-538846105, 1273642420, 1935803229, -1359637233, 996406379, 1201263688, 423734973, 415968277, -1477388697, -232646534, -1477492269, -1718094633, -1898016437, -175024693, 1572714584, -714216075, 434285668, -1533875352, 893664920, 648061059, -2059073898, -1905197771, 2018247426, 953477464, 1427830252, 1883569566, -1235494106, -2144138878, -1366551360, 2114032572,
	}
	var mockedVals []int
	for _, v := range pythonVals {
		mockedVals = append(mockedVals, int(v+2147483647))
	}
	import_rand := &numpyMockSource{values: mockedVals}

	cfg := nn.NNDescentConfig{
		K:             kd.NNeighbors,
		Rng:           import_rand,
		Angular:       false,
		MaxCandidates: 60,
	}

	result := nn.NNDescent(kd.X, distance.Euclidean, cfg)
	indices := result.Indices
	distances := result.Distances

	n := len(kd.X)
	k := kd.NNeighbors

	if len(indices) != n || len(distances) != n {
		t.Fatalf("result shape: got %dx?, want %dx%d", len(indices), n, k)
	}

	// Due to float32 vs float64 differences between Python and Go, the exact path
	// taken by NN-Descent diverges slightly (affecting ~1.5% of edges).
	// We measure the recall against the Python NN-Descent output to ensure parity.
	correctCount := 0
	totalCount := n * k

	for i := range n {
		wantSet := make(map[int]bool)
		for j := range k {
			wantSet[int(kd.KNNIndices[i][j])] = true
		}

		for j := range k {
			if wantSet[indices[i][j]] {
				correctCount++
			}
		}
	}

	recall := float64(correctCount) / float64(totalCount)
	t.Logf("NNDescent recall vs Python NNDescent: %.4f", recall)

	if recall < 0.98 {
		t.Errorf("recall too low: got %.4f, want >= 0.98", recall)
	}
}

type numpyMockSource struct {
	values []int
	idx    int
}

func (s *numpyMockSource) Intn(n int) int {
	if s.idx >= len(s.values) {
		panic("numpyMockSource: ran out of values")
	}
	v := s.values[s.idx]
	s.idx++
	return v
}

func (s *numpyMockSource) Float64() float64                    { return 0 }
func (s *numpyMockSource) NormFloat64() float64                { return 0 }
func (s *numpyMockSource) Perm(n int) []int                    { return nil }
func (s *numpyMockSource) UniformFloat64(l, h float64) float64 { return 0 }
