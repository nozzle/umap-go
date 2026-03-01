package umap

// fuzzy.go implements the fuzzy simplicial set construction pipeline:
//   - SmoothKNNDist: binary search for per-point bandwidths (sigma, rho)
//   - ComputeMembershipStrengths: sparse fuzzy set from kNN
//   - FuzzySimplicialSet: full pipeline from kNN to symmetrized graph
//
// Corresponds to umap_.py smooth_knn_dist(), compute_membership_strengths(),
// fuzzy_simplicial_set().

import (
	"math"

	"github.com/nozzle/umap-go/distance"
	"github.com/nozzle/umap-go/nn"
	umaprand "github.com/nozzle/umap-go/rand"
	"github.com/nozzle/umap-go/sparse"
)

// SmoothKNNResult holds the output of SmoothKNNDist.
type SmoothKNNResult struct {
	Sigmas []float64 // per-point bandwidth parameters
	Rhos   []float64 // per-point nearest neighbor distances
}

// SmoothKNNDist computes the smooth nearest-neighbor distance parameters
// (sigma and rho) for each point using binary search.
//
// For each point, we find sigma such that:
//
//	sum(exp(-(d_i - rho) / sigma)) = log2(k)
//
// where d_i are the distances to k nearest neighbors and rho is the
// distance to the nearest neighbor (local connectivity adjustment).
//
// Matches umap_.py smooth_knn_dist().
func SmoothKNNDist(knnDists [][]float64, k float64, localConnectivity float64) *SmoothKNNResult {
	n := len(knnDists)
	target := math.Log2(k)

	sigmas := make([]float64, n)
	rhos := make([]float64, n)

	meanDistances := 0.0
	for i := range n {
		for _, d := range knnDists[i] {
			meanDistances += d
		}
	}
	if n > 0 && len(knnDists[0]) > 0 {
		meanDistances /= float64(n * len(knnDists[0]))
	}

	for i := range n {
		rhos[i] = computeRho(knnDists[i], localConnectivity)
		sigmas[i] = smoothKNNDistSingle(knnDists[i], rhos[i], target, meanDistances)
	}

	return &SmoothKNNResult{Sigmas: sigmas, Rhos: rhos}
}

// computeRho computes rho for a single point: the distance to the nearest
// neighbor, adjusted by local_connectivity.
//
// With local_connectivity=1.0 (the default), rho is simply the distance
// to the first non-zero nearest neighbor.
func computeRho(dists []float64, localConnectivity float64) float64 {
	nonZeroDists := make([]float64, 0, len(dists))
	for _, d := range dists {
		if d > 0 {
			nonZeroDists = append(nonZeroDists, d)
		}
	}

	if len(nonZeroDists) >= int(math.Floor(localConnectivity)) {
		index := int(math.Floor(localConnectivity)) - 1
		interpolation := localConnectivity - math.Floor(localConnectivity)
		if index < 0 {
			index = 0
		}
		if index >= len(nonZeroDists) {
			return nonZeroDists[len(nonZeroDists)-1]
		}
		if interpolation > 0 && index+1 < len(nonZeroDists) {
			return nonZeroDists[index] + interpolation*(nonZeroDists[index+1]-nonZeroDists[index])
		}
		return nonZeroDists[index]
	}
	if len(nonZeroDists) > 0 {
		return nonZeroDists[len(nonZeroDists)-1] // max of non-zero dists
	}
	return 0
}

// smoothKNNDistSingle performs binary search for sigma for a single point.
func smoothKNNDistSingle(dists []float64, rho, target, meanDist float64) float64 {
	lo := 0.0
	hi := math.Inf(1)
	mid := 1.0

	const nIter = 64
	const bandwidth = 1.0

	for iter := range nIter {
		_ = iter
		val := 0.0
		for j := 1; j < len(dists); j++ {
			d := dists[j] - rho
			if d > 0 {
				val += math.Exp(-d / (mid * bandwidth))
			} else {
				val += 1.0
			}
		}

		if math.Abs(val-target) < 1e-5 {
			break
		}

		if val > target {
			hi = mid
			mid = (lo + hi) / 2.0
		} else {
			lo = mid
			if math.IsInf(hi, 1) {
				mid *= 2
			} else {
				mid = (lo + hi) / 2.0
			}
		}
	}

	if rho > 0 {
		if mid < 1e-3*meanDist {
			mid = 1e-3 * meanDist
		}
	} else {
		if mid < 1e-3*meanDist {
			mid = 1e-3 * meanDist
		}
	}

	return mid
}

// ComputeMembershipStrengths computes the sparse fuzzy set (COO matrix)
// from kNN indices, distances, sigmas, and rhos.
//
// For each point i and neighbor j:
//
//	w_ij = exp(-(d_ij - rho_i) / sigma_i)
//
// Matches umap_.py compute_membership_strengths().
func ComputeMembershipStrengths(knnIndices [][]int, knnDists [][]float64,
	sigmas, rhos []float64, nSamples int) *sparse.COO {

	nNeighbors := len(knnIndices[0])
	coo := sparse.NewCOO(nSamples, nSamples)

	for i := range nSamples {
		for j := range nNeighbors {
			idx := knnIndices[i][j]
			if idx == -1 || idx == i {
				continue
			}

			d := knnDists[i][j]
			var val float64

			if d-rhos[i] <= 0 || sigmas[i] == 0 {
				val = 1.0
			} else {
				val = math.Exp(-(d - rhos[i]) / sigmas[i])
			}

			if val < 1e-300 { // avoid underflow to zero (matches Python MIN_FLOAT)
				val = 1e-300
			}

			coo.Set(i, idx, val)
		}
	}

	return coo
}

// FuzzySimplicialSetResult holds the output of FuzzySimplicialSet.
type FuzzySimplicialSetResult struct {
	Graph       *sparse.CSR     // the symmetrized fuzzy simplicial set graph
	Sigmas      []float64       // per-point sigma values
	Rhos        []float64       // per-point rho values
	SearchIndex *nn.SearchIndex // the nearest neighbor search index
}

// FuzzySimplicialSet constructs the fuzzy simplicial set from raw data.
// This is the main graph construction pipeline:
//
// 1. Compute kNN (brute-force or NN-Descent)
// 2. Smooth kNN distances to get sigma, rho
// 3. Compute membership strengths → sparse matrix
// 4. Symmetrize via fuzzy set union: W + W^T - W*W^T
// 5. Reset local connectivity
//
// Matches umap_.py fuzzy_simplicial_set().
func FuzzySimplicialSet(
	data [][]float64,
	nNeighbors int,
	rng umaprand.Source,
	metric string,
	metricKwds map[string]any,
	localConnectivity float64,
	setOpMixRatio float64,
) *FuzzySimplicialSetResult {
	n := len(data)

	// Get distance function
	distFunc := distance.Named(metric)
	if distFunc == nil {
		if metric == "categorical" {
			distFunc = func(x, y []float64) float64 {
				if len(x) > 0 && len(y) > 0 && x[0] == y[0] {
					return 0
				}
				return 1
			}
		}
	}
	if distFunc == nil {
		// Try parameterized
		paramFunc := distance.NamedWithParam(metric)
		if paramFunc != nil {
			distFunc = func(x, y []float64) float64 {
				return paramFunc(x, y, metricKwds)
			}
		}
	}
	if distFunc == nil {
		distFunc = distance.Euclidean // fallback
	}

	angular := metric == "cosine" || metric == "correlation"

	// Step 1: Compute kNN
	searchIndex := nn.NearestNeighbors(data, nNeighbors, distFunc, rng, angular)
	knnIndices := searchIndex.Indices
	knnDists := searchIndex.Distances

	// Step 2: Smooth kNN distances
	smooth := SmoothKNNDist(knnDists, float64(nNeighbors), localConnectivity)

	// Step 3: Compute membership strengths
	coo := ComputeMembershipStrengths(knnIndices, knnDists, smooth.Sigmas, smooth.Rhos, n)

	// Step 4: Convert to CSR and symmetrize
	graph := coo.ToCSR()
	graphT := graph.Transpose()

	// Fuzzy set union: result = setOpMixRatio * (W + W^T - W*W^T) + (1 - setOpMixRatio) * (W * W^T)
	graph = sparse.FuzzySetUnion(graph, graphT, setOpMixRatio)

	// Step 5: Reset local connectivity
	graph = sparse.ResetLocalConnectivity(graph)

	// Eliminate zeros
	graph = graph.Eliminate(0)

	return &FuzzySimplicialSetResult{
		Graph:       graph,
		Sigmas:      smooth.Sigmas,
		Rhos:        smooth.Rhos,
		SearchIndex: searchIndex,
	}
}

// ComputeMembershipStrengthsBipartite is like ComputeMembershipStrengths but for out-of-sample mapping.
// It creates a bipartite graph of shape (nQueries x nTrain).
func ComputeMembershipStrengthsBipartite(knnIndices [][]int, knnDists [][]float64,
	sigmas, rhos []float64, nQueries, nTrain int) *sparse.COO {

	nNeighbors := len(knnIndices[0])
	coo := sparse.NewCOO(nQueries, nTrain)

	for i := range nQueries {
		for j := range nNeighbors {
			idx := knnIndices[i][j]
			if idx < 0 {
				continue
			}

			d := knnDists[i][j]
			var val float64

			if d-rhos[i] <= 0 || sigmas[i] == 0 {
				val = 1.0
			} else {
				val = math.Exp(-(d - rhos[i]) / sigmas[i])
			}

			if val < 1e-300 {
				val = 1e-300
			}

			coo.Set(i, idx, val)
		}
	}

	return coo
}
