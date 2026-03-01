package sparse

import (
	"sort"
)

// COO is a coordinate-format sparse matrix.
// Duplicate entries are summed on conversion to CSR.
type COO struct {
	Rows, Cols int
	Row        []int
	Col        []int
	Data       []float64
}

// NewCOO creates an empty COO matrix with the given dimensions.
func NewCOO(rows, cols int) *COO {
	return &COO{Rows: rows, Cols: cols}
}

// Set adds a value at (row, col). Duplicate entries are allowed and will
// be summed when converting to CSR.
func (c *COO) Set(row, col int, val float64) {
	c.Row = append(c.Row, row)
	c.Col = append(c.Col, col)
	c.Data = append(c.Data, val)
}

// NNZ returns the number of stored entries (including duplicates).
func (c *COO) NNZ() int {
	return len(c.Data)
}

// ToCSR converts the COO matrix to CSR format.
// Duplicate entries at the same (row, col) are summed.
// Within each row, entries are sorted by column index (matching scipy behavior).
func (c *COO) ToCSR() *CSR {
	nnz := len(c.Data)
	if nnz == 0 {
		indptr := make([]int, c.Rows+1)
		return &CSR{
			Rows:    c.Rows,
			Cols:    c.Cols,
			Indptr:  indptr,
			Indices: nil,
			Data:    nil,
		}
	}

	// Sort by (row, col) to match scipy's COO→CSR conversion order.
	type entry struct {
		row, col int
		val      float64
	}
	entries := make([]entry, nnz)
	for i := range nnz {
		entries[i] = entry{c.Row[i], c.Col[i], c.Data[i]}
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].row != entries[j].row {
			return entries[i].row < entries[j].row
		}
		return entries[i].col < entries[j].col
	})

	// Deduplicate: sum values at same (row, col).
	deduped := make([]entry, 0, nnz)
	deduped = append(deduped, entries[0])
	for i := 1; i < nnz; i++ {
		last := &deduped[len(deduped)-1]
		if entries[i].row == last.row && entries[i].col == last.col {
			last.val += entries[i].val
		} else {
			deduped = append(deduped, entries[i])
		}
	}

	// Build CSR arrays.
	indptr := make([]int, c.Rows+1)
	indices := make([]int, len(deduped))
	data := make([]float64, len(deduped))
	for i, e := range deduped {
		indices[i] = e.col
		data[i] = e.val
		indptr[e.row+1]++
	}
	// Cumulative sum for indptr.
	for i := 1; i <= c.Rows; i++ {
		indptr[i] += indptr[i-1]
	}

	return &CSR{
		Rows:    c.Rows,
		Cols:    c.Cols,
		Indptr:  indptr,
		Indices: indices,
		Data:    data,
	}
}
