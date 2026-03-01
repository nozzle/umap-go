package umap

// spectral_test.go tests SpectralLayout against Python reference data.
// Due to eigenvector sign ambiguity, we test subspace equivalence rather
// than exact element-wise match.

import (
	"math"
	"testing"

	"github.com/nozzle/umap-go/sparse"
)

type spectralLayoutData struct {
	NComponents int         `json:"n_components"`
	Init        [][]float64 `json:"init"`
	Graph       struct {
		Shape   []int     `json:"shape"`
		Indptr  []int     `json:"indptr"`
		Indices []int     `json:"indices"`
		Data    []float64 `json:"data"`
	} `json:"graph"`
}

func TestSpectralLayoutSubspace(t *testing.T) {
	// Load spectral reference (includes the exact graph Python used)
	var td spectralLayoutData
	loadJSON(t, "spectral_layout/iris.json", &td)

	// Build the CSR graph from the reference data so we test SpectralLayout
	// on the exact same graph Python used, eliminating kNN differences.
	graph := &sparse.CSR{
		Rows:    td.Graph.Shape[0],
		Cols:    td.Graph.Shape[1],
		Indptr:  td.Graph.Indptr,
		Indices: td.Graph.Indices,
		Data:    td.Graph.Data,
	}

	embedding := SpectralLayout(graph, td.NComponents)

	if len(embedding) != len(td.Init) {
		t.Fatalf("embedding rows: got %d, want %d", len(embedding), len(td.Init))
	}
	if len(embedding[0]) != td.NComponents {
		t.Fatalf("embedding cols: got %d, want %d", len(embedding[0]), td.NComponents)
	}

	// Check that the embedding spans a similar subspace.
	// Since eigenvectors can be sign-flipped or rotated within degenerate eigenspaces,
	// we test that the pairwise distance structure is preserved.
	// Compute pairwise distance correlation between Go and Python embeddings.
	n := len(embedding)
	goDistances := make([]float64, 0, n*(n-1)/2)
	pyDistances := make([]float64, 0, n*(n-1)/2)

	for i := range n {
		for j := i + 1; j < n; j++ {
			goDist := 0.0
			pyDist := 0.0
			for d := 0; d < td.NComponents; d++ {
				gd := embedding[i][d] - embedding[j][d]
				pd := td.Init[i][d] - td.Init[j][d]
				goDist += gd * gd
				pyDist += pd * pd
			}
			goDistances = append(goDistances, math.Sqrt(goDist))
			pyDistances = append(pyDistances, math.Sqrt(pyDist))
		}
	}

	// Compute Pearson correlation of pairwise distances
	corr := pearsonCorrelation(goDistances, pyDistances)
	t.Logf("pairwise distance correlation: %.6f", corr)

	// The spectral embeddings should have high correlation (>0.9)
	// in their pairwise distance structure
	if corr < 0.9 {
		t.Errorf("pairwise distance correlation too low: %.6f (want > 0.9)", corr)
	}
}

func TestSpectralLayoutDisconnected(t *testing.T) {
	// Test with a disconnected graph (multiple components)
	n := 20
	coo := sparse.NewCOO(n, n)

	// Two separate cliques: 0-9 and 10-19
	for i := range 10 {
		for j := range 10 {
			if i != j {
				coo.Set(i, j, 1.0)
			}
		}
	}
	for i := 10; i < 20; i++ {
		for j := 10; j < 20; j++ {
			if i != j {
				coo.Set(i, j, 1.0)
			}
		}
	}

	graph := coo.ToCSR()
	embedding := SpectralLayout(graph, 2)

	if len(embedding) != n {
		t.Fatalf("embedding rows: got %d, want %d", len(embedding), n)
	}

	// Points from different components should be far apart
	meanIntra := 0.0
	countIntra := 0
	meanInter := 0.0
	countInter := 0

	for i := range n {
		for j := i + 1; j < n; j++ {
			d := 0.0
			for k := range embedding[i] {
				diff := embedding[i][k] - embedding[j][k]
				d += diff * diff
			}
			d = math.Sqrt(d)

			sameComponent := (i < 10 && j < 10) || (i >= 10 && j >= 10)
			if sameComponent {
				meanIntra += d
				countIntra++
			} else {
				meanInter += d
				countInter++
			}
		}
	}

	if countIntra > 0 {
		meanIntra /= float64(countIntra)
	}
	if countInter > 0 {
		meanInter /= float64(countInter)
	}

	t.Logf("mean intra-component distance: %.4f, mean inter-component distance: %.4f",
		meanIntra, meanInter)

	if meanInter <= meanIntra {
		t.Errorf("inter-component distance (%.4f) should be larger than intra-component distance (%.4f)",
			meanInter, meanIntra)
	}
}

func TestComputeMetaEmbeddingFallbackUsesComponentStructure(t *testing.T) {
	const dim = 2
	const nComps = 5 // triggers nComps > 2*dim fallback

	coo := sparse.NewCOO(17, 17)
	addUndirected := func(i, j int) {
		coo.Set(i, j, 1.0)
		coo.Set(j, i, 1.0)
	}

	// Component 0: path of 3 nodes
	addUndirected(0, 1)
	addUndirected(1, 2)

	// Component 1: clique of 4 nodes
	for i := 3; i <= 6; i++ {
		for j := i + 1; j <= 6; j++ {
			addUndirected(i, j)
		}
	}

	// Component 2: edge of 2 nodes
	addUndirected(7, 8)

	// Component 3: star of 5 nodes
	for j := 10; j <= 13; j++ {
		addUndirected(9, j)
	}

	// Component 4: path of 3 nodes (same structure as component 0)
	addUndirected(14, 15)
	addUndirected(15, 16)

	graph := coo.ToCSR()
	gotComps, labels := sparse.ConnectedComponents(graph)
	if gotComps != nComps {
		t.Fatalf("connected components: got %d, want %d", gotComps, nComps)
	}

	meta1 := computeMetaEmbedding(graph, labels, gotComps, dim)
	meta2 := computeMetaEmbedding(graph, labels, gotComps, dim)

	if len(meta1) != nComps || len(meta1[0]) != dim {
		t.Fatalf("meta embedding shape: got %dx%d, want %dx%d", len(meta1), len(meta1[0]), nComps, dim)
	}

	d04 := euclideanDistance(meta1[0], meta1[4]) // identical component structures
	d01 := euclideanDistance(meta1[0], meta1[1])
	d03 := euclideanDistance(meta1[0], meta1[3])
	if d04 >= d01 || d04 >= d03 {
		t.Fatalf("expected similar components to be closer: d04=%.6f d01=%.6f d03=%.6f", d04, d01, d03)
	}

	// Deterministic fallback: pairwise distances should match across calls.
	for i := range nComps {
		for j := i + 1; j < nComps; j++ {
			d1 := euclideanDistance(meta1[i], meta1[j])
			d2 := euclideanDistance(meta2[i], meta2[j])
			if math.Abs(d1-d2) > 1e-12 {
				t.Fatalf("pairwise distance mismatch for (%d,%d): %.12f vs %.12f", i, j, d1, d2)
			}
		}
	}
}

// pearsonCorrelation computes the Pearson correlation coefficient.
func pearsonCorrelation(x, y []float64) float64 {
	n := float64(len(x))
	var sumX, sumY, sumXY, sumX2, sumY2 float64
	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	num := n*sumXY - sumX*sumY
	den := math.Sqrt((n*sumX2 - sumX*sumX) * (n*sumY2 - sumY*sumY))
	if den == 0 {
		return 0
	}
	return num / den
}

func euclideanDistance(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}
