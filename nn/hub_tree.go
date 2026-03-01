package nn

import (
	"math"
)

// MIN_SPLIT_BALANCE is the minimum acceptable balance for a hub-based split.
// If a split is more skewed than this, a leaf is created instead.
const minSplitBalance = 0.1

// computeGlobalDegrees computes the global in-degree for all points in the graph.
func computeGlobalDegrees(neighborIndices [][]int, nPoints int) []int {
	degrees := make([]int, nPoints)
	for i := 0; i < len(neighborIndices); i++ {
		for j := 0; j < len(neighborIndices[i]); j++ {
			neighbor := neighborIndices[i][j]
			if neighbor >= 0 && neighbor < nPoints {
				degrees[neighbor]++
			}
		}
	}
	return degrees
}

// getTopKHubIndices gets the indices of the top k highest-degree points from a subset.
func getTopKHubIndices(indices []int, globalDegrees []int, k int) []int {
	nPoints := len(indices)
	actualK := k
	if nPoints < k {
		actualK = nPoints
	}

	topDegrees := make([]int, actualK)
	for i := range topDegrees {
		topDegrees[i] = -1
	}
	topIndices := make([]int, actualK)

	for i := 0; i < nPoints; i++ {
		deg := globalDegrees[indices[i]]

		if deg > topDegrees[actualK-1] {
			insertPos := actualK - 1
			for insertPos > 0 && deg > topDegrees[insertPos-1] {
				insertPos--
			}

			// Shift elements down
			for j := actualK - 1; j > insertPos; j-- {
				topDegrees[j] = topDegrees[j-1]
				topIndices[j] = topIndices[j-1]
			}

			// Insert new element
			topDegrees[insertPos] = deg
			topIndices[insertPos] = indices[i]
		}
	}

	return topIndices
}

func euclideanHubSplit(data [][]float64, indices []int, globalDegrees []int, rngState *TauRandState) ([]int, []int, []float64, float64, float64) {
	dim := len(data[0])
	nPoints := len(indices)

	topHubs := getTopKHubIndices(indices, globalDegrees, 3)
	nHubs := len(topHubs)

	bestBalance := 0.0
	bestNLeft := 0
	bestNRight := 0
	bestHyperplane := make([]float64, dim)
	bestOffset := 0.0
	bestSide := make([]int8, nPoints)
	side := make([]int8, nPoints)

	hyperplaneVector := make([]float64, dim)

	for hi := 0; hi < nHubs; hi++ {
		for hj := hi + 1; hj < nHubs; hj++ {
			left := topHubs[hi]
			right := topHubs[hj]

			hyperplaneOffset := 0.0
			for d := 0; d < dim; d++ {
				hyperplaneVector[d] = data[left][d] - data[right][d]
				hyperplaneOffset -= hyperplaneVector[d] * (data[left][d] + data[right][d]) / 2.0
			}

			nLeft := 0
			nRight := 0

			for i := 0; i < nPoints; i++ {
				margin := hyperplaneOffset
				for d := 0; d < dim; d++ {
					margin += hyperplaneVector[d] * data[indices[i]][d]
				}

				if margin > 1e-8 {
					side[i] = 0
					nLeft++
				} else if margin < -1e-8 {
					side[i] = 1
					nRight++
				} else {
					side[i] = int8(i % 2)
					if side[i] == 0 {
						nLeft++
					} else {
						nRight++
					}
				}
			}

			if nLeft == 0 || nRight == 0 {
				continue
			}

			balance := float64(minInt(nLeft, nRight)) / float64(nPoints)

			if balance > bestBalance {
				bestBalance = balance
				bestNLeft = nLeft
				bestNRight = nRight
				bestOffset = hyperplaneOffset
				copy(bestHyperplane, hyperplaneVector)
				copy(bestSide, side)
			}
		}
	}

	if bestNLeft == 0 || bestNRight == 0 {
		bestNLeft = 0
		bestNRight = 0
		for i := 0; i < nPoints; i++ {
			bestSide[i] = int8(math.Abs(float64(TauRandInt(rngState)))) % 2
			if bestSide[i] == 0 {
				bestNLeft++
			} else {
				bestNRight++
			}
		}
	}

	indicesLeft := make([]int, 0, bestNLeft)
	indicesRight := make([]int, 0, bestNRight)

	for i := 0; i < nPoints; i++ {
		if bestSide[i] == 0 {
			indicesLeft = append(indicesLeft, indices[i])
		} else {
			indicesRight = append(indicesRight, indices[i])
		}
	}

	return indicesLeft, indicesRight, bestHyperplane, bestOffset, bestBalance
}

func angularHubSplit(data [][]float64, indices []int, globalDegrees []int, rngState *TauRandState) ([]int, []int, []float64, float64, float64) {
	dim := len(data[0])
	nPoints := len(indices)

	topHubs := getTopKHubIndices(indices, globalDegrees, 3)
	nHubs := len(topHubs)

	bestBalance := 0.0
	bestNLeft := 0
	bestNRight := 0
	bestHyperplane := make([]float64, dim)
	bestOffset := 0.0
	bestSide := make([]int8, nPoints)
	side := make([]int8, nPoints)

	hyperplaneVector := make([]float64, dim)

	for hi := 0; hi < nHubs; hi++ {
		for hj := hi + 1; hj < nHubs; hj++ {
			left := topHubs[hi]
			right := topHubs[hj]

			var leftNorm, rightNorm float64
			for d := 0; d < dim; d++ {
				leftNorm += data[left][d] * data[left][d]
				rightNorm += data[right][d] * data[right][d]
			}
			leftNorm = math.Sqrt(leftNorm)
			rightNorm = math.Sqrt(rightNorm)

			if leftNorm == 0 || rightNorm == 0 {
				if dim > 0 {
					for d := range dim {
						hyperplaneVector[d] = 0
					}
					hyperplaneVector[0] = 1.0
				}
			} else {
				for d := 0; d < dim; d++ {
					hyperplaneVector[d] = (data[left][d] / leftNorm) - (data[right][d] / rightNorm)
				}
			}

			nLeft := 0
			nRight := 0

			for i := 0; i < nPoints; i++ {
				margin := 0.0
				for d := 0; d < dim; d++ {
					margin += hyperplaneVector[d] * data[indices[i]][d]
				}

				if margin > 1e-8 {
					side[i] = 0
					nLeft++
				} else if margin < -1e-8 {
					side[i] = 1
					nRight++
				} else {
					side[i] = int8(i % 2)
					if side[i] == 0 {
						nLeft++
					} else {
						nRight++
					}
				}
			}

			if nLeft == 0 || nRight == 0 {
				continue
			}

			balance := float64(minInt(nLeft, nRight)) / float64(nPoints)

			if balance > bestBalance {
				bestBalance = balance
				bestNLeft = nLeft
				bestNRight = nRight
				bestOffset = 0.0
				copy(bestHyperplane, hyperplaneVector)
				copy(bestSide, side)
			}
		}
	}

	if bestNLeft == 0 || bestNRight == 0 {
		bestNLeft = 0
		bestNRight = 0
		for i := 0; i < nPoints; i++ {
			bestSide[i] = int8(math.Abs(float64(TauRandInt(rngState)))) % 2
			if bestSide[i] == 0 {
				bestNLeft++
			} else {
				bestNRight++
			}
		}
	}

	indicesLeft := make([]int, 0, bestNLeft)
	indicesRight := make([]int, 0, bestNRight)

	for i := 0; i < nPoints; i++ {
		if bestSide[i] == 0 {
			indicesLeft = append(indicesLeft, indices[i])
		} else {
			indicesRight = append(indicesRight, indices[i])
		}
	}

	return indicesLeft, indicesRight, bestHyperplane, bestOffset, bestBalance
}

func buildHubTreeRecursive(tree *FlatTree, data [][]float64, indices []int, neighborIndices [][]int, globalDegrees []int, leafSize int, rngState *TauRandState, angular bool, maxDepth int) int {
	nodeID := len(tree.Children)

	if len(indices) <= leafSize || maxDepth <= 0 {
		leaf := make([]int, len(indices))
		copy(leaf, indices)
		tree.Hyperplanes = append(tree.Hyperplanes, nil)
		tree.Offsets = append(tree.Offsets, 0)
		tree.Children = append(tree.Children, [2]int{-1, -1})
		tree.Indices = append(tree.Indices, leaf)
		return nodeID
	}

	var leftIndices, rightIndices []int
	var hyperplane []float64
	var offset, balance float64

	if angular {
		leftIndices, rightIndices, hyperplane, offset, balance = angularHubSplit(data, indices, globalDegrees, rngState)
	} else {
		leftIndices, rightIndices, hyperplane, offset, balance = euclideanHubSplit(data, indices, globalDegrees, rngState)
	}

	if balance < minSplitBalance {
		leaf := make([]int, len(indices))
		copy(leaf, indices)
		tree.Hyperplanes = append(tree.Hyperplanes, nil)
		tree.Offsets = append(tree.Offsets, 0)
		tree.Children = append(tree.Children, [2]int{-1, -1})
		tree.Indices = append(tree.Indices, leaf)
		return nodeID
	}

	tree.Hyperplanes = append(tree.Hyperplanes, hyperplane)
	tree.Offsets = append(tree.Offsets, offset)
	tree.Children = append(tree.Children, [2]int{-1, -1})
	tree.Indices = append(tree.Indices, nil)

	leftChild := buildHubTreeRecursive(tree, data, leftIndices, neighborIndices, globalDegrees, leafSize, rngState, angular, maxDepth-1)
	rightChild := buildHubTreeRecursive(tree, data, rightIndices, neighborIndices, globalDegrees, leafSize, rngState, angular, maxDepth-1)

	tree.Children[nodeID] = [2]int{leftChild, rightChild}

	return nodeID
}

// MakeHubTree builds a hub-based RP-tree.
func MakeHubTree(data [][]float64, neighborIndices [][]int, leafSize int, rngState *TauRandState, angular bool, maxDepth int) *FlatTree {
	globalDegrees := computeGlobalDegrees(neighborIndices, len(data))

	indices := make([]int, len(data))
	for i := range indices {
		indices[i] = i
	}

	tree := &FlatTree{
		Hyperplanes: make([][]float64, 0),
		Offsets:     make([]float64, 0),
		Children:    make([][2]int, 0),
		Indices:     make([][]int, 0),
	}

	buildHubTreeRecursive(tree, data, indices, neighborIndices, globalDegrees, leafSize, rngState, angular, maxDepth)
	return tree
}
