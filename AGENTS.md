# AGENTS.md

This document serves as context and instructions for AI agents and future contributors working on the `umap-go` repository. 

## Project Goal
Build a complete, highly-accurate, pure-Go port of the Python UMAP library (`umap-learn` pinned to `v0.5.11`). The overarching objective is ~100% test coverage and **exact mathematical parity** across all distance metrics, nearest-neighbor structures, and UMAP algorithmic stages.

## Strict Mandates & Development Rules

1. **Byte-for-Byte Reproducibility:** 
   We aim for exact parity with Python. If Python's implementation contains mathematical quirks or specific floating-point behaviors, we must replicate them in Go to guarantee parity. 
2. **Float32 vs Float64 Allowances:** 
   Go defaults to `float64` while Numba/PyNNDescent strictly uses `float32`. Distance sorting and numeric accumulation will diverge slightly. When building tests, measure success via tight error bounds or recall metrics (e.g., >99% recall on NNDescent indexes) rather than requiring strict boolean equality for floats.
3. **PRNG Synchronization:**
   UMAP and NNDescent are highly stochastic. We use a custom `umaprand` package and a 3-element `int64` XOR-shift Tausworthe PRNG (`nn.TauRandState`). To test parity, RNG seeds and sequences must be meticulously matched to Python's Numba states. **Do not modify PRNG advancement without verifying against Python outputs.**
4. **No Heavy C-Bindings:** 
   The library must remain pure Go. Avoid porting features that require massive C-library wrappers unless explicitly approved (e.g., `InverseTransform` is currently deferred because it relies on the `QHull` C library for N-dimensional Delaunay triangulation).

## Architecture Context

* **`distance/`**: Contains all distance metrics. Tested against dense pairwise, gradient, and sparse Python reference outputs.
* **`nn/`**: The PyNNDescent port. Contains `NNDescent`, RP-Forest construction, graph-informed Hub Trees (used for out-of-sample queries), and min-max Heap structures.
* **`umap.go`**: Core algorithms. 
  * `Fit()` constructs the fuzzy simplicial set and optimizes the Euclidean layout. 
  * `Transform()` uses `nn.SearchIndex` to query the previously built NN graph, limits edges via `DisconnectionDistance`, calculates bipartite membership strengths, and optimizes the new points while keeping training embeddings frozen (`move_other = false`).

## Testing Strategy
Python scripts in `testdata/` (e.g., `generate.py`) are used to freeze intermediate UMAP states (pairwise distances, knn indices, sigma/rho smoothing, etc.) into JSON/CSV formats. Go tests then read these fixtures and assert that our Go implementation produces the exact same internal tensors.
