package umap

// spectral.go implements spectral initialization for UMAP embeddings.
//
// Constructs a normalized graph Laplacian from the fuzzy simplicial set
// and computes the smallest non-trivial eigenvectors to use as the initial
// embedding coordinates.
//
// Corresponds to umap/spectral.py spectral_layout().
//
// Current implementation uses dense eigendecomposition via gonum mat.EigenSym,
// which works well for n <= ~8000.
//
// TODO: Implement Lanczos sparse eigensolver for n > ~8000
// TODO: Implement LOBPCG for n > ~2M
// TODO: Implement tswspectral fallback

import (
	"math"

	"gonum.org/v1/gonum/mat"

	"github.com/nozzle/umap-go/sparse"
)

// SpectralLayout computes the spectral initialization embedding.
//
// Algorithm:
// 1. Compute the normalized graph Laplacian: L = D^{-1/2} (D - A) D^{-1/2}
// 2. Find the smallest nComponents+1 eigenvectors
// 3. Discard the trivial eigenvector (constant, eigenvalue ~0)
// 4. Return the next nComponents eigenvectors, scaled by sqrt(eigenvalue)
//
// If the graph has multiple connected components, each component is
// handled independently and results are combined.
//
// graph: the symmetrized fuzzy simplicial set (CSR)
// nComponents: number of embedding dimensions
func SpectralLayout(graph *sparse.CSR, nComponents int) [][]float64 {
	n := graph.Rows

	// Check for multiple connected components
	nComps, labels := sparse.ConnectedComponents(graph)
	if nComps > 1 {
		return multiComponentSpectral(graph, nComponents, nComps, labels)
	}

	return singleComponentSpectral(graph, n, nComponents)
}

// singleComponentSpectral handles the single-component case.
// Returns raw eigenvectors (no min/max normalization), matching Python's
// _spectral_layout which just returns eigenvectors[:, order].
func singleComponentSpectral(graph *sparse.CSR, n, nComponents int) [][]float64 {
	// Build normalized Laplacian as a dense matrix
	L := buildNormalizedLaplacian(graph, n)

	// Eigendecomposition
	var eigen mat.EigenSym
	ok := eigen.Factorize(L, true)
	if !ok {
		// Fallback to random initialization
		return randomInit(n, nComponents)
	}

	values := eigen.Values(nil)
	var vectors mat.Dense
	eigen.VectorsTo(&vectors)

	// Find the first non-trivial eigenvector.
	// For a connected graph there is exactly one zero eigenvalue.
	// Eigenvalues from EigenSym are sorted ascending.
	// Skip all eigenvalues that are essentially zero (< threshold).
	threshold := 1e-8
	startIdx := 0
	for startIdx < len(values) && math.Abs(values[startIdx]) < threshold {
		startIdx++
	}
	if startIdx == 0 {
		startIdx = 1 // always skip at least the smallest
	}

	embedding := make([][]float64, n)
	for i := range n {
		embedding[i] = make([]float64, nComponents)
	}

	for d := range nComponents {
		eigIdx := startIdx + d
		if eigIdx >= len(values) {
			break
		}
		for i := range n {
			embedding[i][d] = vectors.At(i, eigIdx)
		}
	}

	return embedding
}

// buildNormalizedLaplacian constructs the dense normalized graph Laplacian.
// L = I - D^{-1/2} A D^{-1/2}
func buildNormalizedLaplacian(graph *sparse.CSR, n int) *mat.SymDense {
	// Compute degree vector D
	d := make([]float64, n)
	for i := range n {
		for j := graph.Indptr[i]; j < graph.Indptr[i+1]; j++ {
			d[i] += graph.Data[j]
		}
	}

	// D^{-1/2}
	dInvSqrt := make([]float64, n)
	for i := range n {
		if d[i] > 0 {
			dInvSqrt[i] = 1.0 / math.Sqrt(d[i])
		}
	}

	// Build symmetric matrix: L = I - D^{-1/2} A D^{-1/2}
	L := mat.NewSymDense(n, nil)

	// Diagonal: 1.0 (from I)
	for i := range n {
		L.SetSym(i, i, 1.0)
	}

	// Off-diagonal: -D^{-1/2}[i] * A[i,j] * D^{-1/2}[j]
	for i := range n {
		for idx := graph.Indptr[i]; idx < graph.Indptr[i+1]; idx++ {
			j := graph.Indices[idx]
			val := graph.Data[idx]
			normalized := -dInvSqrt[i] * val * dInvSqrt[j]
			if i == j {
				// Diagonal: 1 - D^{-1/2}[i] * A[i,i] * D^{-1/2}[i]
				L.SetSym(i, i, L.At(i, i)+normalized)
			} else if i < j {
				// Only set upper triangle (symmetric)
				L.SetSym(i, j, normalized)
			}
		}
	}

	return L
}

// multiComponentSpectral handles graphs with multiple connected components.
// Matches Python's multi_component_layout():
//  1. Compute a meta-embedding for each component's centroid.
//     For nComps <= 2*dim, use a fixed layout based on identity/negated identity.
//  2. For each component, run spectral layout on the subgraph, scale it, and
//     position it at the component's meta-embedding centroid.
func multiComponentSpectral(graph *sparse.CSR, dim, nComps int, labels []int) [][]float64 {
	n := graph.Rows
	embedding := make([][]float64, n)
	for i := range n {
		embedding[i] = make([]float64, dim)
	}

	// Step 1: Compute meta-embedding for component centroids.
	// Python: for nComps <= 2*dim, k = ceil(nComps/2), base = hstack(eye(k), zeros(k, dim-k)),
	// meta = vstack(base, -base)[:nComps]
	metaEmbedding := computeMetaEmbedding(nComps, dim)

	// Step 2: For each component, run spectral layout and position.
	for comp := range nComps {
		// Gather indices for this component
		var compIndices []int
		for i, label := range labels {
			if label == comp {
				compIndices = append(compIndices, i)
			}
		}

		compN := len(compIndices)
		center := metaEmbedding[comp]

		// Compute data_range: min distance from this component's centroid to others, /2
		dataRange := computeDataRange(metaEmbedding, comp)

		if compN < 2*dim || compN <= dim+1 {
			// Too small for spectral: place uniformly around centroid
			// Python uses random_state.uniform(-data_range, data_range, size=(n, dim)) + center
			for localIdx, globalIdx := range compIndices {
				for d := range dim {
					// Deterministic spread within data_range
					t := float64(localIdx) / float64(compN)
					embedding[globalIdx][d] = center[d] + dataRange*(2*t-1)
				}
			}
			continue
		}

		// Extract subgraph
		subgraph := extractSubgraph(graph, compIndices)

		// Spectral layout on subgraph
		subEmbedding := singleComponentSpectral(subgraph, compN, dim)

		// Scale: expansion = data_range / max(abs(component_embedding))
		maxAbs := 0.0
		for i := range compN {
			for d := range dim {
				a := math.Abs(subEmbedding[i][d])
				if a > maxAbs {
					maxAbs = a
				}
			}
		}
		expansion := 1.0
		if maxAbs > 0 {
			expansion = dataRange / maxAbs
		}

		// Place back: component_embedding * expansion + center
		for localIdx, globalIdx := range compIndices {
			for d := range dim {
				embedding[globalIdx][d] = subEmbedding[localIdx][d]*expansion + center[d]
			}
		}
	}

	return embedding
}

// computeMetaEmbedding creates the meta-embedding for component centroids.
// For nComps <= 2*dim: k = ceil(nComps/2), base = [eye(k) | zeros(k, dim-k)],
// meta = [base; -base][:nComps]
// For nComps > 2*dim: TODO — use component_layout with spectral embedding of centroids
func computeMetaEmbedding(nComps, dim int) [][]float64 {
	if nComps > 2*dim {
		// TODO: Implement component_layout for many components
		// For now, fall back to simple linear spacing
		meta := make([][]float64, nComps)
		for c := range nComps {
			meta[c] = make([]float64, dim)
			meta[c][0] = float64(c) * 10.0
		}
		return meta
	}

	k := (nComps + 1) / 2 // ceil(nComps / 2)
	// base is k x dim: eye(k) padded with zeros
	base := make([][]float64, k)
	for i := range k {
		base[i] = make([]float64, dim)
		if i < dim {
			base[i][i] = 1.0
		}
	}

	// meta = [base; -base][:nComps]
	meta := make([][]float64, nComps)
	for i := range nComps {
		meta[i] = make([]float64, dim)
		if i < k {
			copy(meta[i], base[i])
		} else {
			for d := range dim {
				meta[i][d] = -base[i-k][d]
			}
		}
	}

	return meta
}

// computeDataRange computes half the minimum distance from component comp
// to all other components in the meta-embedding.
func computeDataRange(metaEmbedding [][]float64, comp int) float64 {
	minDist := math.Inf(1)
	center := metaEmbedding[comp]

	for i := range metaEmbedding {
		if i == comp {
			continue
		}
		d := 0.0
		for k := range center {
			diff := center[k] - metaEmbedding[i][k]
			d += diff * diff
		}
		d = math.Sqrt(d)
		if d > 0 && d < minDist {
			minDist = d
		}
	}

	if math.IsInf(minDist, 1) {
		return 1.0
	}
	return minDist / 2.0
}

// extractSubgraph creates a subgraph from the given indices.
func extractSubgraph(graph *sparse.CSR, indices []int) *sparse.CSR {
	n := len(indices)

	// Build global→local index mapping
	globalToLocal := make(map[int]int)
	for localIdx, globalIdx := range indices {
		globalToLocal[globalIdx] = localIdx
	}

	coo := sparse.NewCOO(n, n)
	for localI, globalI := range indices {
		for idx := graph.Indptr[globalI]; idx < graph.Indptr[globalI+1]; idx++ {
			globalJ := graph.Indices[idx]
			localJ, ok := globalToLocal[globalJ]
			if ok {
				coo.Set(localI, localJ, graph.Data[idx])
			}
		}
	}

	return coo.ToCSR()
}

// normalizeEmbedding centers and scales the embedding to a reasonable range.
func normalizeEmbedding(embedding [][]float64) {
	n := len(embedding)
	if n == 0 {
		return
	}
	nComponents := len(embedding[0])

	// Expand coordinates by a small random amount to break ties
	expansion := 10.0 / math.Sqrt(float64(n))

	for d := range nComponents {
		// Find range
		minVal := math.Inf(1)
		maxVal := math.Inf(-1)
		for i := range n {
			v := embedding[i][d]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}

		rng := maxVal - minVal
		if rng == 0 {
			rng = 1
		}

		// Center and scale
		for i := range n {
			embedding[i][d] = expansion * (embedding[i][d] - minVal) / rng
		}
	}
}

// randomInit generates a random initial embedding.
func randomInit(n, nComponents int) [][]float64 {
	embedding := make([][]float64, n)
	// Simple deterministic "random" init based on index
	for i := range n {
		embedding[i] = make([]float64, nComponents)
		for d := range nComponents {
			// Spread out uniformly in a small range
			embedding[i][d] = 10.0 * float64(i) / float64(n)
		}
	}
	return embedding
}
