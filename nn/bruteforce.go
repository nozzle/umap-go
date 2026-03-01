package nn

import (
	"math"
	"sort"

	"github.com/nozzle/umap-go/distance"
)

// BruteForceKNN computes exact k-nearest neighbors using brute-force
// pairwise distance computation. This is used when n_samples < 4096.
//
// The Python equivalent is fast_knn_indices() in umap_.py which uses
// argpartition, plus a full pairwise distance matrix.
//
// Returns (indices, distances) arrays of shape [n][k].
func BruteForceKNN(data [][]float64, k int, distFunc distance.Func) ([][]int, [][]float64) {
	n := len(data)

	// Compute full pairwise distance matrix
	D := make([][]float64, n)
	for i := range n {
		D[i] = make([]float64, n)
		for j := range n {
			if i != j {
				D[i][j] = distFunc(data[i], data[j])
			}
		}
	}

	return FastKNNIndices(D, k)
}

// FastKNNIndices finds the k nearest neighbors from a precomputed distance
// matrix D. Matches UMAP v0.5.11's fast_knn_indices() which uses a
// deterministic stable sort for reproducibility.
//
// Returns (indices, distances) arrays of shape [n][k].
func FastKNNIndices(D [][]float64, k int) ([][]int, [][]float64) {
	n := len(D)
	indices := make([][]int, n)
	distances := make([][]float64, n)

	for i := range n {
		// Create index array and sort by distance (stable sort for tie-breaking)
		order := make([]int, n)
		for j := range n {
			order[j] = j
		}
		row := D[i]
		sort.SliceStable(order, func(a, b int) bool {
			return row[order[a]] < row[order[b]]
		})

		// Take first k+1, skip self (which should be at position 0 with distance 0)
		indices[i] = make([]int, k)
		distances[i] = make([]float64, k)
		pos := 0
		for _, j := range order {
			if pos >= k {
				break
			}
			// v0.5.11: includes self in the output (at position 0)
			// The caller decides whether to skip self.
			indices[i][pos] = j
			distances[i][pos] = row[j]
			pos++
		}
	}

	return indices, distances
}

// PairwiseDistances computes the full n x n pairwise distance matrix.
func PairwiseDistances(data [][]float64, distFunc distance.Func) [][]float64 {
	n := len(data)
	D := make([][]float64, n)
	for i := range n {
		D[i] = make([]float64, n)
		for j := range n {
			if i != j {
				D[i][j] = distFunc(data[i], data[j])
			}
		}
	}
	return D
}

// InitFromBruteForce initializes a Heap from brute-force kNN results.
// This converts the sorted output of BruteForceKNN into a max-heap.
func InitFromBruteForce(indices [][]int, distances [][]float64) *Heap {
	n := len(indices)
	k := len(indices[0])
	h := NewHeap(n, k)

	for i := range n {
		for j := range k {
			h.Indices[i][j] = indices[i][j]
			h.Distances[i][j] = distances[i][j]
			h.Flags[i][j] = true
		}
		// Heapify: build max-heap from bottom up
		for pos := k/2 - 1; pos >= 0; pos-- {
			heapifySiftDown(h.Distances[i], h.Indices[i], h.Flags[i], pos, k)
		}
	}

	return h
}

func heapifySiftDown(dists []float64, inds []int, flags []bool, pos, n int) {
	for {
		left := 2*pos + 1
		right := 2*pos + 2
		largest := pos

		if left < n && dists[left] > dists[largest] {
			largest = left
		}
		if right < n && dists[right] > dists[largest] {
			largest = right
		}

		if largest == pos {
			break
		}

		dists[pos], dists[largest] = dists[largest], dists[pos]
		inds[pos], inds[largest] = inds[largest], inds[pos]
		flags[pos], flags[largest] = flags[largest], flags[pos]
		pos = largest
	}
}

// InitFromRandom initializes a Heap with random neighbors.
// This is used to seed NN-Descent when RP-trees are not available.
func InitFromRandom(n, k int, distFunc distance.Func, data [][]float64, rngState *TauRandState) *Heap {
	h := NewHeap(n, k)

	for i := range n {
		for range k {
			idx := TauRandIntRange(rngState, n)
			if idx == i {
				continue
			}
			d := distFunc(data[i], data[idx])
			if d < math.Inf(1) {
				h.Push(i, idx, d, true)
			}
		}
	}

	return h
}
