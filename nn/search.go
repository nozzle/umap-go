package nn

import (
	"container/heap"
	"math"

	"github.com/nozzle/umap-go/distance"
	umaprand "github.com/nozzle/umap-go/rand"
)

// SearchIndex holds the state required to query a fitted NNDescent graph.
type SearchIndex struct {
	Data      [][]float64
	Indices   [][]int
	Forest    RPForest
	DistFunc  distance.Func
	RngState  TauRandState
	Distances [][]float64
	isBruteForce bool
}

// NewSearchIndex creates a SearchIndex from the output of NNDescent.
func NewSearchIndex(data [][]float64, indices [][]int, forest RPForest, distFunc distance.Func, rng umaprand.Source) *SearchIndex {
	return &SearchIndex{
		Data:     data,
		Indices:  indices,
		Forest:   forest,
		DistFunc: distFunc,
		RngState: makeTauRandState(rng),
	}
}

// Query searches the index for the k nearest neighbors of each point in queryData.
// epsilon controls the trade-off between accuracy and search cost.
func (idx *SearchIndex) Query(queryData [][]float64, k int, epsilon float64) ([][]int, [][]float64) {
	if idx.isBruteForce {
		// Just do brute force
		return BruteForceKNNQuery(idx.Data, queryData, k, idx.DistFunc)
	}

	nQueries := len(queryData)
	outIndices := make([][]int, nQueries)
	outDists := make([][]float64, nQueries)

	minDistance := 0.0

	for i, q := range queryData {
		resultHeap := NewHeap(1, k)
		internalRng := idx.RngState // copy to ensure reproducible search

		// 1. Init from Tree
		var candidates []int
		if len(idx.Forest) > 0 {
			tree := idx.Forest[0]
			candidates = treeSearch(tree, q, &internalRng)
		}

		visited := make(map[int]bool)
		
		// seedSet is a min-heap of (distance, index)
		seedSet := &minDistanceHeap{}
		heap.Init(seedSet)

		for _, cand := range candidates {
			if !visited[cand] {
				visited[cand] = true
				d := idx.DistFunc(q, idx.Data[cand])
				resultHeap.Push(0, cand, d, true)
				heap.Push(seedSet, distItem{d: d, index: cand})
			}
		}

		// 2. Random samples if needed
		nPoints := len(idx.Data)
		if seedSet.Len() < k {
			for j := 0; j < k && len(visited) < nPoints; j++ {
				cand := int(math.Abs(float64(TauRandInt(&internalRng)))) % nPoints
				if !visited[cand] {
					visited[cand] = true
					d := idx.DistFunc(q, idx.Data[cand])
					resultHeap.Push(0, cand, d, true)
					heap.Push(seedSet, distItem{d: d, index: cand})
				}
			}
		}

		// 3. Search graph
		if seedSet.Len() > 0 {
			worstDist := resultHeap.Distances[0][0]
			distanceBound := worstDist + epsilon*(worstDist-minDistance)

			for seedSet.Len() > 0 {
				item := heap.Pop(seedSet).(distItem)
				if item.d >= distanceBound {
					break // seedSet is a min-heap, so remaining are >= distanceBound
				}

				vertex := item.index
				for _, neighbor := range idx.Indices[vertex] {
					if neighbor < 0 || visited[neighbor] {
						continue
					}
					visited[neighbor] = true

					d := idx.DistFunc(q, idx.Data[neighbor])
					if d < distanceBound {
						resultHeap.Push(0, neighbor, d, true)
						heap.Push(seedSet, distItem{d: d, index: neighbor})
						
						worstDist = resultHeap.Distances[0][0]
						distanceBound = worstDist + epsilon*(worstDist-minDistance)
					}
				}
			}
		}

		resultHeap.Deheap()
		outIndices[i] = resultHeap.Indices[0]
		outDists[i] = resultHeap.Distances[0]
	}

	return outIndices, outDists
}

func selectSide(hyperplane []float64, offset float64, point []float64, rngState *TauRandState) int {
	margin := offset
	for d := range hyperplane {
		margin += hyperplane[d] * point[d]
	}
	if math.Abs(margin) < 1e-8 {
		side := int(math.Abs(float64(TauRandInt(rngState)))) % 2
		if side == 0 {
			return 0
		}
		return 1
	}
	if margin <= 0 {
		return 0 // leftChild
	}
	return 1 // rightChild
}

func treeSearch(tree *FlatTree, point []float64, rngState *TauRandState) []int {
	node := 0
	for tree.Children[node][0] != -1 {
		side := selectSide(tree.Hyperplanes[node], tree.Offsets[node], point, rngState)
		if side == 0 {
			node = tree.Children[node][0]
		} else {
			node = tree.Children[node][1]
		}
	}
	return tree.Indices[node]
}

type distItem struct {
	d     float64
	index int
}

type minDistanceHeap []distItem

func (h minDistanceHeap) Len() int           { return len(h) }
func (h minDistanceHeap) Less(i, j int) bool { return h[i].d < h[j].d }
func (h minDistanceHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minDistanceHeap) Push(x any)        { *h = append(*h, x.(distItem)) }
func (h *minDistanceHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}
