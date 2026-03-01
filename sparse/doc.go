// Package sparse provides COO and CSR sparse matrix types for UMAP.
//
// These are intentionally simple implementations covering only the operations
// UMAP needs: construction, conversion, transpose, and fuzzy set operations.
// We implement these ourselves rather than using an external library to maintain
// precise control over element ordering for reproducibility with the Python
// reference implementation.
package sparse
