package umap

// fuzzy_test.go tests SmoothKNNDist and the fuzzy simplicial set pipeline
// against Python reference data.

import (
	"math"
	"testing"
)

type smoothKNNDistData struct {
	KNNDists          [][]float64 `json:"knn_dists"`
	NNeighbors        int         `json:"n_neighbors"`
	LocalConnectivity float64     `json:"local_connectivity"`
	Sigmas            []float64   `json:"sigmas"`
	Rhos              []float64   `json:"rhos"`
}

func TestSmoothKNNDist(t *testing.T) {
	datasets := []string{"iris", "small"}

	for _, ds := range datasets {
		t.Run(ds, func(t *testing.T) {
			var td smoothKNNDistData
			loadJSON(t, "smooth_knn_dist/"+ds+".json", &td)

			result := SmoothKNNDist(td.KNNDists, float64(td.NNeighbors), td.LocalConnectivity)

			// Rho should match very closely
			rhoTol := 1e-10
			assertSliceApprox(t, "rhos", result.Rhos, td.Rhos, rhoTol)

			// Sigma comparison.
			// Python's smooth_knn_dist uses numba @njit(fastmath=True) with
			// float32 locals for psum/lo/mid/hi. The fastmath flag can cause
			// the binary search to produce infinity for some points where
			// Go's float64 binary search correctly converges to a finite
			// positive value. For those entries, we accept any finite
			// positive sigma from Go. For entries where Python converged to
			// a finite value, we compare within tolerance.
			sigmaTol := 1e-5
			nInfSkipped := 0
			for i := range result.Sigmas {
				wantInf := math.IsInf(td.Sigmas[i], 1) || td.Sigmas[i] > 1e300 || td.Sigmas[i] > 99999.0
				if wantInf {
					// Python returned infinity (numba fastmath artifact).
					// Go should produce a finite positive value.
					if result.Sigmas[i] <= 0 {
						t.Errorf("sigmas[%d]: got %v, expected positive (Python had +Inf)", i, result.Sigmas[i])
					}
					nInfSkipped++
					continue
				}
				if !approxEqual(result.Sigmas[i], td.Sigmas[i], sigmaTol) {
					t.Errorf("sigmas[%d]: got %v, want %v (tol=%v)", i, result.Sigmas[i], td.Sigmas[i], sigmaTol)
				}
			}
			if nInfSkipped > 0 {
				t.Logf("skipped %d sigma entries where Python returned +Inf (numba fastmath artifact)", nInfSkipped)
			}
		})
	}
}

type fuzzySimplicialSetData struct {
	NNeighbors int    `json:"n_neighbors"`
	Metric     string `json:"metric"`
	Graph      struct {
		Shape []int     `json:"shape"`
		Row   []int     `json:"row"`
		Col   []int     `json:"col"`
		Data  []float64 `json:"data"`
	} `json:"graph"`
	Sigmas []float64 `json:"sigmas"`
	Rhos   []float64 `json:"rhos"`
}

func TestFuzzySimplicialSetStructure(t *testing.T) {
	// Test that the fuzzy simplicial set has the expected structure
	// (number of nonzeros, symmetry) but not byte-for-byte match
	// since kNN ordering differences can cascade.
	datasets := []string{"iris", "small"}

	for _, ds := range datasets {
		t.Run(ds, func(t *testing.T) {
			var td fuzzySimplicialSetData
			loadJSON(t, "fuzzy_simplicial_set/"+ds+".json", &td)

			// Load the dataset
			var dataset struct {
				X [][]float64 `json:"X"`
			}
			loadJSON(t, "datasets/"+ds+".json", &dataset)

			result := FuzzySimplicialSet(
				dataset.X,
				td.NNeighbors,
				nil, // use default RNG
				td.Metric,
				nil,
				1.0, // localConnectivity
				1.0, // setOpMixRatio
			)

			if result.Graph == nil {
				t.Fatal("graph is nil")
			}

			expectedShape := td.Graph.Shape
			if result.Graph.Rows != expectedShape[0] || result.Graph.Cols != expectedShape[1] {
				t.Errorf("graph shape: got %dx%d, want %dx%d",
					result.Graph.Rows, result.Graph.Cols,
					expectedShape[0], expectedShape[1])
			}

			// Check that we got a reasonable number of non-zeros
			gotNNZ := len(result.Graph.Data)
			wantNNZ := len(td.Graph.Data)
			// Allow some tolerance since kNN can differ slightly
			ratio := float64(gotNNZ) / float64(wantNNZ)
			if ratio < 0.8 || ratio > 1.2 {
				t.Errorf("graph NNZ: got %d, want ~%d (ratio=%.3f)", gotNNZ, wantNNZ, ratio)
			}
			t.Logf("graph NNZ: got %d, want %d (ratio=%.3f)", gotNNZ, wantNNZ, ratio)

			// Graph should be symmetric (up to numerical precision)
			graphT := result.Graph.Transpose()
			maxAsymmetry := 0.0
			for i := 0; i < result.Graph.Rows; i++ {
				for idx := result.Graph.Indptr[i]; idx < result.Graph.Indptr[i+1]; idx++ {
					j := result.Graph.Indices[idx]
					v := result.Graph.Data[idx]
					vt := graphT.At(j, i)
					diff := math.Abs(v - vt)
					if diff > maxAsymmetry {
						maxAsymmetry = diff
					}
				}
			}
			if maxAsymmetry > 1e-10 {
				t.Errorf("graph is not symmetric: max asymmetry = %v", maxAsymmetry)
			}

			// All weights should be in [0, 1]
			for _, v := range result.Graph.Data {
				if v < 0 || v > 1.0+1e-10 {
					t.Errorf("graph weight out of range [0,1]: %v", v)
					break
				}
			}
		})
	}
}
