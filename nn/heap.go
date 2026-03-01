package nn

import "math"

// Heap implements a max-heap for k-nearest neighbor search.
//
// For each of N data points, it maintains a heap of K neighbors, storing
// their indices, distances, and a flag indicating if the neighbor is new.
// This matches the heap structure in pynndescent (indices, distances, flags).
//
// The heap is a max-heap by distance: the farthest neighbor is at the root,
// so we can quickly check if a candidate is closer than the current worst.
type Heap struct {
	N         int         // number of data points
	K         int         // number of neighbors per point
	Indices   [][]int     // [N][K] neighbor indices (-1 = empty)
	Distances [][]float64 // [N][K] neighbor distances (+Inf = empty)
	Flags     [][]bool    // [N][K] true = new (not yet processed by NN-Descent)
}

// NewHeap creates a new heap for n data points with k neighbors each.
func NewHeap(n, k int) *Heap {
	h := &Heap{
		N:         n,
		K:         k,
		Indices:   make([][]int, n),
		Distances: make([][]float64, n),
		Flags:     make([][]bool, n),
	}
	for i := range n {
		h.Indices[i] = make([]int, k)
		h.Distances[i] = make([]float64, k)
		h.Flags[i] = make([]bool, k)
		for j := range k {
			h.Indices[i][j] = -1
			h.Distances[i][j] = math.Inf(1)
		}
	}
	return h
}

// Push attempts to add a neighbor (index j, distance d) to the heap for
// point i. Returns true if the neighbor was added (i.e., it was closer
// than the current farthest neighbor). The isNew flag is set on the entry.
//
// Matches pynndescent heap_push().
func (h *Heap) Push(i, j int, d float64, isNew bool) bool {
	if d >= h.Distances[i][0] {
		return false
	}

	// Check for duplicate
	for k := range h.K {
		if h.Indices[i][k] == j {
			return false
		}
	}

	// Replace root (farthest) with new neighbor
	h.Distances[i][0] = d
	h.Indices[i][0] = j
	h.Flags[i][0] = isNew

	// Sift down to restore max-heap property
	h.siftDown(i, 0)
	return true
}

// Unchecked push — does not check for duplicates. Used during initialization.
// Matches pynndescent unchecked_heap_push().
func (h *Heap) UncheckedPush(i, j int, d float64, isNew bool) bool {
	if d >= h.Distances[i][0] {
		return false
	}

	h.Distances[i][0] = d
	h.Indices[i][0] = j
	h.Flags[i][0] = isNew

	h.siftDown(i, 0)
	return true
}

// MaxDist returns the maximum distance (root) for point i.
func (h *Heap) MaxDist(i int) float64 {
	return h.Distances[i][0]
}

// siftDown maintains the max-heap property for point i starting at position pos.
func (h *Heap) siftDown(i, pos int) {
	k := h.K
	dists := h.Distances[i]
	inds := h.Indices[i]
	flags := h.Flags[i]

	for {
		left := 2*pos + 1
		right := 2*pos + 2
		largest := pos

		if left < k && dists[left] > dists[largest] {
			largest = left
		}
		if right < k && dists[right] > dists[largest] {
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

// Deheap sorts each row of the heap by distance (ascending) in-place.
// After this, Indices[i][0] is the nearest neighbor of point i.
// This destroys the heap property but is needed for final output.
// Matches pynndescent deheap_sort().
func (h *Heap) Deheap() {
	for i := range h.N {
		h.deheapRow(i)
	}
}

// deheapRow sorts a single row by extracting elements from the max-heap.
func (h *Heap) deheapRow(i int) {
	dists := h.Distances[i]
	inds := h.Indices[i]
	flags := h.Flags[i]
	n := h.K

	for end := n - 1; end > 0; end-- {
		// Swap root (max) with end
		dists[0], dists[end] = dists[end], dists[0]
		inds[0], inds[end] = inds[end], inds[0]
		flags[0], flags[end] = flags[end], flags[0]

		// Sift down in reduced heap [0, end)
		pos := 0
		for {
			left := 2*pos + 1
			right := 2*pos + 2
			largest := pos

			if left < end && dists[left] > dists[largest] {
				largest = left
			}
			if right < end && dists[right] > dists[largest] {
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
}

// BuildCandidates separates new and old neighbors for NN-Descent.
// Returns (newCandidates, oldCandidates) heaps.
// Matches pynndescent new_build_candidates().
func (h *Heap) BuildCandidates(maxCandidates int, rngState *TauRandState) (newCands, oldCands *Heap) {
	newCands = NewHeap(h.N, maxCandidates)
	oldCands = NewHeap(h.N, maxCandidates)

	for i := range h.N {
		for j := range h.K {
			idx := h.Indices[i][j]
			if idx < 0 {
				continue
			}
			d := TauRand(rngState)
			if h.Flags[i][j] {
				newCands.UncheckedPush(i, idx, d, true)
				newCands.UncheckedPush(idx, i, d, true)
				h.Flags[i][j] = false
			} else {
				oldCands.UncheckedPush(i, idx, d, true)
				oldCands.UncheckedPush(idx, i, d, true)
			}
		}
	}

	return newCands, oldCands
}
