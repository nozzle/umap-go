package nn

import (
	"math"

	"github.com/nozzle/umap-go/distance"
	umaprand "github.com/nozzle/umap-go/rand"
)

// RPTree represents a random projection tree for approximate nearest neighbor
// candidate generation. Matches pynndescent's RP-tree construction.
//
// Internal (non-leaf) nodes store a hyperplane defined by (hyperplane, offset).
// Leaf nodes store a list of data point indices.
type RPTree struct {
	Hyperplanes [][]float64 // per-node hyperplane normal vectors (nil for leaves)
	Offsets     []float64   // per-node offset values
	Children    [][2]int    // per-node [left, right] child indices (-1 for leaves)
	Indices     [][]int     // per-node leaf indices (nil for internal nodes)
}

// FlatTree is a flattened representation of an RP-tree, storing just the
// leaf membership for efficient queries.
type FlatTree struct {
	Hyperplanes [][]float64 // per-node hyperplane normal vectors
	Offsets     []float64   // per-node offsets
	Children    [][2]int    // per-node children
	Indices     [][]int     // per-node leaf indices
}

// RPForest is a collection of RP-trees.
type RPForest []*FlatTree

// MakeForest builds a forest of RP-trees.
// nTrees: number of trees to build.
// leafSize: maximum leaf size.
// rng: random source for tree construction.
// data: the dataset (n_samples x n_features).
// angular: if true, use angular (cosine-based) random projections.
func MakeForest(data [][]float64, nTrees int, leafSize int, rng umaprand.Source, angular bool) RPForest {
	n := len(data)
	forest := make(RPForest, nTrees)

	// Generate rng states for each tree
	treeRngStates := make([]TauRandState, nTrees)
	for t := range nTrees {
		treeRngStates[t] = makeTauRandState(rng)
	}

	for t := range nTrees {
		// Build indices list [0, 1, ..., n-1]
		indices := make([]int, n)
		for i := range n {
			indices[i] = i
		}

		tree := buildRPTree(data, indices, leafSize, &treeRngStates[t], angular)
		forest[t] = flattenTree(tree)
		_ = t
	}

	return forest
}

// buildRPTree recursively builds an RP-tree.
func buildRPTree(data [][]float64, indices []int, leafSize int, rngState *TauRandState, angular bool) *RPTree {
	tree := &RPTree{}
	buildRPTreeRecursive(tree, data, indices, leafSize, rngState, angular)
	return tree
}

// buildRPTreeRecursive builds the tree by recursively splitting nodes.
func buildRPTreeRecursive(tree *RPTree, data [][]float64, indices []int, leafSize int, rngState *TauRandState, angular bool) int {
	nodeID := len(tree.Hyperplanes)

	if len(indices) <= leafSize {
		// Leaf node
		leaf := make([]int, len(indices))
		copy(leaf, indices)
		tree.Hyperplanes = append(tree.Hyperplanes, nil)
		tree.Offsets = append(tree.Offsets, 0)
		tree.Children = append(tree.Children, [2]int{-1, -1})
		tree.Indices = append(tree.Indices, leaf)
		return nodeID
	}

	// Pick two random distinct points to define the hyperplane
	var leftIdx, rightIdx int
	for {
		leftIdx = TauRandIntRange(rngState, len(indices))
		rightIdx = TauRandIntRange(rngState, len(indices))
		if leftIdx != rightIdx {
			break
		}
	}
	left := data[indices[leftIdx]]
	right := data[indices[rightIdx]]

	var hyperplane []float64
	var offset float64

	if angular {
		hyperplane, offset = angularRandomProjectionSplit(left, right)
	} else {
		hyperplane, offset = euclideanRandomProjectionSplit(left, right, rngState)
	}

	// Partition indices by which side of the hyperplane they fall on
	leftIndices := make([]int, 0, len(indices)/2)
	rightIndices := make([]int, 0, len(indices)/2)

	for _, idx := range indices {
		margin := offset
		for d := range hyperplane {
			margin += hyperplane[d] * data[idx][d]
		}
		if margin <= 0 {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}

	// Handle degenerate splits: if one side is empty, split evenly
	if len(leftIndices) == 0 || len(rightIndices) == 0 {
		mid := len(indices) / 2
		leftIndices = indices[:mid]
		rightIndices = indices[mid:]
	}

	// Allocate this node
	tree.Hyperplanes = append(tree.Hyperplanes, hyperplane)
	tree.Offsets = append(tree.Offsets, offset)
	tree.Children = append(tree.Children, [2]int{-1, -1})
	tree.Indices = append(tree.Indices, nil)

	// Recurse
	leftChild := buildRPTreeRecursive(tree, data, leftIndices, leafSize, rngState, angular)
	rightChild := buildRPTreeRecursive(tree, data, rightIndices, leafSize, rngState, angular)
	tree.Children[nodeID] = [2]int{leftChild, rightChild}

	return nodeID
}

// euclideanRandomProjectionSplit creates a hyperplane between two points.
// The hyperplane normal is (left - right), and a random point along the
// midline determines the offset. Matches pynndescent euclidean_random_projection_split.
func euclideanRandomProjectionSplit(left, right []float64, rngState *TauRandState) ([]float64, float64) {
	dim := len(left)
	hyperplane := make([]float64, dim)
	midpoint := make([]float64, dim)

	for d := range dim {
		hyperplane[d] = left[d] - right[d]
		midpoint[d] = (left[d] + right[d]) / 2.0
	}

	offset := 0.0
	for d := range dim {
		offset -= hyperplane[d] * midpoint[d]
	}

	return hyperplane, offset
}

// angularRandomProjectionSplit creates a hyperplane for angular (cosine) distance.
// The hyperplane normal is the normalized difference of the L2-normalized vectors.
func angularRandomProjectionSplit(left, right []float64) ([]float64, float64) {
	dim := len(left)
	hyperplane := make([]float64, dim)

	// Normalize both vectors
	var normLeft, normRight float64
	for d := range dim {
		normLeft += left[d] * left[d]
		normRight += right[d] * right[d]
	}
	normLeft = math.Sqrt(normLeft)
	normRight = math.Sqrt(normRight)

	if normLeft == 0 || normRight == 0 {
		// Degenerate: return arbitrary hyperplane
		if dim > 0 {
			hyperplane[0] = 1
		}
		return hyperplane, 0
	}

	for d := range dim {
		hyperplane[d] = left[d]/normLeft - right[d]/normRight
	}

	// Normalize the hyperplane
	var normHP float64
	for d := range dim {
		normHP += hyperplane[d] * hyperplane[d]
	}
	normHP = math.Sqrt(normHP)
	if normHP > 0 {
		for d := range dim {
			hyperplane[d] /= normHP
		}
	}

	return hyperplane, 0 // offset is 0 for angular
}

// flattenTree converts an RPTree to a FlatTree.
func flattenTree(tree *RPTree) *FlatTree {
	return &FlatTree{
		Hyperplanes: tree.Hyperplanes,
		Offsets:     tree.Offsets,
		Children:    tree.Children,
		Indices:     tree.Indices,
	}
}

// GetLeafArray returns all leaf arrays from the forest, concatenated.
// This is used by NN-Descent to initialize candidates.
func GetLeafArray(forest RPForest) [][]int {
	var leaves [][]int
	for _, tree := range forest {
		for i, indices := range tree.Indices {
			if indices != nil && tree.Children[i][0] == -1 {
				leaves = append(leaves, indices)
			}
		}
	}
	return leaves
}

// InitFromForest initializes a Heap from RP-forest leaf co-occurrence.
// For each leaf, all pairs of points in the leaf are considered as candidates.
func InitFromForest(forest RPForest, data [][]float64, k int, distFunc distance.Func) *Heap {
	n := len(data)
	h := NewHeap(n, k)

	leaves := GetLeafArray(forest)
	for _, leaf := range leaves {
		for i := range leaf {
			for j := i + 1; j < len(leaf); j++ {
				a, b := leaf[i], leaf[j]
				d := distFunc(data[a], data[b])
				h.Push(a, b, d, true)
				h.Push(b, a, d, true)
			}
		}
	}

	return h
}
