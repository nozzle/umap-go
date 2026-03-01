# umap-go

`umap-go` is a pure-Go implementation of Uniform Manifold Approximation and Projection (UMAP). 

This project was built with a strict mandate: achieve **exact mathematical parity** with the reference Python `umap-learn` library (pinned to v0.5.11) and its underlying `pynndescent` neighbor search. It is designed to be highly accurate, reproducible, and dependency-free (pure Go, no CGO).

## Features

- **Pure Go:** No reliance on CGO, Cython, or external C libraries.
- **Mathematical Parity:** Built to replicate the exact float math, graph constructions, and algorithmic stages of Python's UMAP, including exact PRNG sequence replication for reproducibility.
- **Complete Fit & Transform:** Supports training a model (`Fit` / `FitTransform`) and mapping new, out-of-sample data into an existing embedding space (`Transform`).
- **Custom NN-Descent:** Includes a fully replicated PyNNDescent implementation, including RP-Trees, graph-informed Hub Trees, and constrained graph search for exact nearest neighbor approximations.
- **Metrics:** Supports all standard UMAP distance metrics (Euclidean, Cosine, etc.).

## Installation

```bash
go get github.com/nozzle/umap-go
```

## Quick Start

```go
package main

import (
	"fmt"
	"github.com/nozzle/umap-go"
)

func main() {
	// 1. Configure UMAP options
	opts := umap.DefaultOptions()
	opts.NNeighbors = 15
	opts.MinDist = 0.1
	opts.NComponents = 2

	// 2. Initialize model
	model := umap.New(opts)

	// 3. Fit data
	var trainingData [][]float64 // Load your n x d data here
	embedding, err := model.FitTransform(trainingData, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Embedded %d points into %d dimensions\n", len(embedding), len(embedding[0]))

	// 4. Transform new, out-of-sample data
	var newData [][]float64 
	newEmbedding, err := model.Transform(newData)
	if err != nil {
		panic(err)
	}
	fmt.Println("Successfully transformed new data points.")
}
```

## Current Status & Roadmap

- [x] Phase 1: Distance metrics (~100% parity against dense, sparse, and gradient reference outputs)
- [x] Phase 2: NN-Descent & RP-Trees (Including Tausworthe PRNG sequence matching)
- [x] Phase 3: Core UMAP Fit, Graph Construction, and SGD Optimization 
- [x] Phase 4: `Transform()` API (Out-of-sample mapping via bipartite graph strengths and frozen SGD)
- [ ] Phase 5: `InverseTransform()` API (Currently deferred. Python relies on N-dimensional Delaunay triangulation via `scipy.spatial.Delaunay` / QHull, which lacks a lightweight pure-Go equivalent).
