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
	for i := 0; i < n; i++ {
		if len(indices[i]) != k {
			t.Fatalf("indices[%d] length: got %d, want %d", i, len(indices[i]), k)
		}
		for j := 0; j < k; j++ {
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
	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
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
	for i := 0; i < n; i++ {
		if D[i][i] != 0 {
			t.Errorf("D[%d][%d] = %v, want 0", i, i, D[i][i])
		}
	}

	// Should be symmetric
	for i := 0; i < n; i++ {
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
	vals := make([]int32, 100)
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
	for i := 0; i < 1000; i++ {
		v := nn.TauRandIntRange(&state3, n)
		if v < 0 || v >= n {
			t.Fatalf("TauRandIntRange out of range: got %d, want [0,%d)", v, n)
		}
	}
}
