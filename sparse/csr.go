package sparse

import (
	"fmt"
	"math"
)

// CSR is a compressed sparse row matrix.
type CSR struct {
	Rows, Cols int
	Indptr     []int     // length Rows+1
	Indices    []int     // column indices
	Data       []float64 // values
}

// NNZ returns the number of stored nonzero entries.
func (c *CSR) NNZ() int {
	return len(c.Data)
}

// At returns the value at (row, col). O(nnz_per_row) lookup.
func (c *CSR) At(row, col int) float64 {
	for j := c.Indptr[row]; j < c.Indptr[row+1]; j++ {
		if c.Indices[j] == col {
			return c.Data[j]
		}
	}
	return 0
}

// RowNNZ returns the number of nonzero entries in the given row.
func (c *CSR) RowNNZ(row int) int {
	return c.Indptr[row+1] - c.Indptr[row]
}

// RowIndices returns the column indices for the given row.
func (c *CSR) RowIndices(row int) []int {
	return c.Indices[c.Indptr[row]:c.Indptr[row+1]]
}

// RowData returns the values for the given row.
func (c *CSR) RowData(row int) []float64 {
	return c.Data[c.Indptr[row]:c.Indptr[row+1]]
}

// Transpose returns a new CSR matrix that is the transpose of this one.
// Corresponds to scipy.sparse.csr_matrix.T.tocsr().
func (c *CSR) Transpose() *CSR {
	coo := NewCOO(c.Cols, c.Rows)
	for i := range c.Rows {
		for j := c.Indptr[i]; j < c.Indptr[i+1]; j++ {
			coo.Set(c.Indices[j], i, c.Data[j])
		}
	}
	return coo.ToCSR()
}

// Copy returns a deep copy of this CSR matrix.
func (c *CSR) Copy() *CSR {
	indptr := make([]int, len(c.Indptr))
	copy(indptr, c.Indptr)
	indices := make([]int, len(c.Indices))
	copy(indices, c.Indices)
	data := make([]float64, len(c.Data))
	copy(data, c.Data)
	return &CSR{
		Rows:    c.Rows,
		Cols:    c.Cols,
		Indptr:  indptr,
		Indices: indices,
		Data:    data,
	}
}

// Eliminate removes entries at or below the given threshold and compacts the storage.
// Eliminate(0) removes entries where data[j] <= 0 (explicit zeros and negatives).
func (c *CSR) Eliminate(threshold float64) *CSR {
	coo := NewCOO(c.Rows, c.Cols)
	for i := range c.Rows {
		for j := c.Indptr[i]; j < c.Indptr[i+1]; j++ {
			if c.Data[j] > threshold {
				coo.Set(i, c.Indices[j], c.Data[j])
			}
		}
	}
	return coo.ToCSR()
}

// MaxVal returns the maximum value in the matrix.
func (c *CSR) MaxVal() float64 {
	if len(c.Data) == 0 {
		return 0
	}
	max := c.Data[0]
	for _, v := range c.Data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// RowSums returns the sum of each row.
func (c *CSR) RowSums() []float64 {
	sums := make([]float64, c.Rows)
	for i := range c.Rows {
		s := 0.0
		for j := c.Indptr[i]; j < c.Indptr[i+1]; j++ {
			s += c.Data[j]
		}
		sums[i] = s
	}
	return sums
}

// MulVec computes y = A*x where A is this CSR matrix.
// x and y must have the correct dimensions. y is overwritten.
func (c *CSR) MulVec(x, y []float64) {
	for i := range c.Rows {
		s := 0.0
		for j := c.Indptr[i]; j < c.Indptr[i+1]; j++ {
			s += c.Data[j] * x[c.Indices[j]]
		}
		y[i] = s
	}
}

// FuzzySetUnion computes the fuzzy set union of two sparse matrices:
//
//	result = setOpMixRatio * (A + B - A*B) + (1 - setOpMixRatio) * (A*B)
//
// where A*B is element-wise multiplication.
// This corresponds to UMAP's symmetrization of the directed kNN graph.
// Both matrices must have the same dimensions.
//
// Corresponds to umap_.py fuzzy_simplicial_set symmetrization.
func FuzzySetUnion(a, b *CSR, setOpMixRatio float64) *CSR {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("sparse.FuzzySetUnion: dimension mismatch: (%d,%d) vs (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	coo := NewCOO(a.Rows, a.Cols)

	for i := range a.Rows {
		// Build maps for row i of both matrices.
		aRow := make(map[int]float64)
		for j := a.Indptr[i]; j < a.Indptr[i+1]; j++ {
			aRow[a.Indices[j]] = a.Data[j]
		}
		bRow := make(map[int]float64)
		for j := b.Indptr[i]; j < b.Indptr[i+1]; j++ {
			bRow[b.Indices[j]] = b.Data[j]
		}

		// Union of column indices.
		allCols := make(map[int]struct{})
		for col := range aRow {
			allCols[col] = struct{}{}
		}
		for col := range bRow {
			allCols[col] = struct{}{}
		}

		for col := range allCols {
			aVal := aRow[col]
			bVal := bRow[col]
			prod := aVal * bVal
			// union = A + B - A*B, intersection = A*B
			val := setOpMixRatio*(aVal+bVal-prod) + (1-setOpMixRatio)*prod
			if val != 0 {
				coo.Set(i, col, val)
			}
		}
	}

	return coo.ToCSR()
}

// ResetLocalConnectivity normalizes each row so the maximum value is 1,
// then symmetrizes via fuzzy union. This ensures each point has at least
// one connection with full strength.
//
// Corresponds to umap_.py reset_local_connectivity().
func ResetLocalConnectivity(graph *CSR) *CSR {
	coo := NewCOO(graph.Rows, graph.Cols)
	for i := range graph.Rows {
		maxVal := 0.0
		for j := graph.Indptr[i]; j < graph.Indptr[i+1]; j++ {
			if graph.Data[j] > maxVal {
				maxVal = graph.Data[j]
			}
		}
		if maxVal > 0 {
			for j := graph.Indptr[i]; j < graph.Indptr[i+1]; j++ {
				coo.Set(i, graph.Indices[j], graph.Data[j]/maxVal)
			}
		}
	}
	normalized := coo.ToCSR()
	trans := normalized.Transpose()
	return FuzzySetUnion(normalized, trans, 1.0)
}

// GeneralSimplicialSetIntersection computes a weighted intersection of
// a simplicial set with a target simplicial set.
//
// result[i,j] = left[i,j] * right[i,j]^weight
//
// When weight=0, returns left unchanged. When weight=1, returns element-wise product.
// Used for supervised UMAP to combine data topology with target topology.
//
// Corresponds to umap_.py general_simplicial_set_intersection().
func GeneralSimplicialSetIntersection(left, right *CSR, weight float64) *CSR {
	if weight == 0 {
		return left.Copy()
	}

	coo := NewCOO(left.Rows, left.Cols)
	for i := range left.Rows {
		// Build map for right row.
		rightRow := make(map[int]float64)
		for j := right.Indptr[i]; j < right.Indptr[i+1]; j++ {
			rightRow[right.Indices[j]] = right.Data[j]
		}

		for j := left.Indptr[i]; j < left.Indptr[i+1]; j++ {
			col := left.Indices[j]
			leftVal := left.Data[j]
			rightVal := rightRow[col] // 0 if not present

			var val float64
			if rightVal > 0 {
				val = leftVal * math.Pow(rightVal, weight)
			}
			// If rightVal is 0, the product is 0 regardless.

			if val > 0 {
				coo.Set(i, col, val)
			}
		}
	}

	return coo.ToCSR()
}

// ConnectedComponents finds connected components in a symmetric sparse graph
// using BFS. Returns the number of components and a label array.
//
// Corresponds to scipy.sparse.csgraph.connected_components().
func ConnectedComponents(graph *CSR) (int, []int) {
	n := graph.Rows
	labels := make([]int, n)
	for i := range labels {
		labels[i] = -1
	}

	nComponents := 0
	queue := make([]int, 0, n)

	for i := range n {
		if labels[i] >= 0 {
			continue
		}
		// BFS from node i.
		labels[i] = nComponents
		queue = queue[:0]
		queue = append(queue, i)

		for len(queue) > 0 {
			node := queue[0]
			queue = queue[1:]

			for j := graph.Indptr[node]; j < graph.Indptr[node+1]; j++ {
				neighbor := graph.Indices[j]
				if graph.Data[j] > 0 && labels[neighbor] < 0 {
					labels[neighbor] = nComponents
					queue = append(queue, neighbor)
				}
			}
		}
		nComponents++
	}

	return nComponents, labels
}
