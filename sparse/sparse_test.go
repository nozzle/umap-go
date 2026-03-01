package sparse_test

import (
	"math"
	"testing"

	"github.com/nozzle/umap-go/sparse"
)

func TestCOOBasic(t *testing.T) {
	coo := sparse.NewCOO(3, 3)
	coo.Set(0, 1, 1.0)
	coo.Set(1, 2, 2.0)
	coo.Set(2, 0, 3.0)

	if coo.NNZ() != 3 {
		t.Errorf("NNZ: got %d, want 3", coo.NNZ())
	}
}

func TestCOOToCSR(t *testing.T) {
	coo := sparse.NewCOO(3, 3)
	coo.Set(0, 1, 1.0)
	coo.Set(0, 2, 2.0)
	coo.Set(1, 0, 3.0)
	coo.Set(2, 1, 4.0)

	csr := coo.ToCSR()

	if csr.Rows != 3 || csr.Cols != 3 {
		t.Errorf("shape: got %dx%d, want 3x3", csr.Rows, csr.Cols)
	}

	// Check values
	tests := []struct {
		i, j int
		want float64
	}{
		{0, 1, 1.0},
		{0, 2, 2.0},
		{1, 0, 3.0},
		{2, 1, 4.0},
		{0, 0, 0.0}, // missing
		{1, 1, 0.0}, // missing
	}

	for _, tt := range tests {
		got := csr.At(tt.i, tt.j)
		if math.Abs(got-tt.want) > 1e-12 {
			t.Errorf("At(%d,%d): got %v, want %v", tt.i, tt.j, got, tt.want)
		}
	}
}

func TestCSRTranspose(t *testing.T) {
	coo := sparse.NewCOO(2, 3)
	coo.Set(0, 1, 1.0)
	coo.Set(0, 2, 2.0)
	coo.Set(1, 0, 3.0)

	csr := coo.ToCSR()
	csrT := csr.Transpose()

	if csrT.Rows != 3 || csrT.Cols != 2 {
		t.Errorf("transposed shape: got %dx%d, want 3x2", csrT.Rows, csrT.Cols)
	}

	// A[0,1]=1 → A^T[1,0]=1
	if got := csrT.At(1, 0); math.Abs(got-1.0) > 1e-12 {
		t.Errorf("At(1,0): got %v, want 1.0", got)
	}
	// A[0,2]=2 → A^T[2,0]=2
	if got := csrT.At(2, 0); math.Abs(got-2.0) > 1e-12 {
		t.Errorf("At(2,0): got %v, want 2.0", got)
	}
	// A[1,0]=3 → A^T[0,1]=3
	if got := csrT.At(0, 1); math.Abs(got-3.0) > 1e-12 {
		t.Errorf("At(0,1): got %v, want 3.0", got)
	}
}

func TestFuzzySetUnion(t *testing.T) {
	// Build a simple graph
	coo := sparse.NewCOO(3, 3)
	coo.Set(0, 1, 0.8)
	coo.Set(1, 2, 0.5)
	coo.Set(2, 0, 0.3)

	graph := coo.ToCSR()
	graphT := graph.Transpose()

	result := sparse.FuzzySetUnion(graph, graphT, 1.0)

	// The result should be symmetric
	for i := range 3 {
		for j := i + 1; j < 3; j++ {
			vij := result.At(i, j)
			vji := result.At(j, i)
			if math.Abs(vij-vji) > 1e-12 {
				t.Errorf("asymmetric: At(%d,%d)=%v, At(%d,%d)=%v", i, j, vij, j, i, vji)
			}
		}
	}

	// Check fuzzy union formula: W + W^T - W*W^T
	// For (0,1): 0.8 + 0 - 0.8*0 = 0.8
	got01 := result.At(0, 1)
	want01 := 0.8
	if math.Abs(got01-want01) > 1e-10 {
		t.Errorf("At(0,1): got %v, want %v", got01, want01)
	}
}

func TestConnectedComponents(t *testing.T) {
	// Two disconnected components: {0,1,2} and {3,4}
	coo := sparse.NewCOO(5, 5)
	coo.Set(0, 1, 1.0)
	coo.Set(1, 0, 1.0)
	coo.Set(1, 2, 1.0)
	coo.Set(2, 1, 1.0)
	coo.Set(3, 4, 1.0)
	coo.Set(4, 3, 1.0)

	graph := coo.ToCSR()
	nComps, labels := sparse.ConnectedComponents(graph)

	if nComps != 2 {
		t.Errorf("nComponents: got %d, want 2", nComps)
	}

	// 0, 1, 2 should have same label
	if labels[0] != labels[1] || labels[1] != labels[2] {
		t.Errorf("component 1 labels: %d %d %d", labels[0], labels[1], labels[2])
	}
	// 3, 4 should have same label
	if labels[3] != labels[4] {
		t.Errorf("component 2 labels: %d %d", labels[3], labels[4])
	}
	// Different components should have different labels
	if labels[0] == labels[3] {
		t.Error("different components should have different labels")
	}
}

func TestResetLocalConnectivity(t *testing.T) {
	coo := sparse.NewCOO(3, 3)
	coo.Set(0, 1, 0.5)
	coo.Set(0, 2, 0.3)
	coo.Set(1, 0, 0.7)
	coo.Set(1, 2, 0.2)
	coo.Set(2, 0, 0.4)
	coo.Set(2, 1, 0.6)

	graph := coo.ToCSR()
	result := sparse.ResetLocalConnectivity(graph)

	// After reset, each row's max should be 1.0
	for i := range 3 {
		maxVal := 0.0
		for idx := result.Indptr[i]; idx < result.Indptr[i+1]; idx++ {
			if result.Data[idx] > maxVal {
				maxVal = result.Data[idx]
			}
		}
		if math.Abs(maxVal-1.0) > 1e-10 {
			t.Errorf("row %d max: got %v, want 1.0", i, maxVal)
		}
	}
}

func TestEliminate(t *testing.T) {
	coo := sparse.NewCOO(2, 2)
	coo.Set(0, 0, 1.0)
	coo.Set(0, 1, 0.0)
	coo.Set(1, 0, 0.5)
	coo.Set(1, 1, 0.0)

	csr := coo.ToCSR()
	result := csr.Eliminate(0)

	// Only non-zero entries should remain
	nnz := 0
	for i := 0; i < result.Rows; i++ {
		for idx := result.Indptr[i]; idx < result.Indptr[i+1]; idx++ {
			nnz++
			if result.Data[idx] == 0 {
				t.Error("zero value should have been eliminated")
			}
		}
	}
	if nnz != 2 {
		t.Errorf("NNZ after eliminate: got %d, want 2", nnz)
	}
}
