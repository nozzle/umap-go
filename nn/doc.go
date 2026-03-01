// Package nn implements nearest neighbor search for UMAP.
//
// It provides:
//   - A max-heap for maintaining k-nearest neighbors
//   - Brute-force pairwise kNN search (used when n < 4096)
//   - Random projection trees for candidate generation
//   - NN-Descent approximate nearest neighbor search
//
// All algorithms match the PyNNDescent library (v0.5.x) used by UMAP v0.5.11.
package nn
