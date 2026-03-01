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

## Capability Matrix

| Capability | Status | Notes |
| --- | --- | --- |
| `Fit(X)` / `FitTransform(X, nil)` | Supported | Unsupervised training and embedding generation. |
| `FitTransform(X, y)` | Supported | Supervised mode (`y []float64`) via target-graph intersection. |
| `Transform(XNew)` | Supported | Out-of-sample mapping after `Fit`/`FitTransform`. |
| `InverseTransform(XEmbedded)` | Deferred | Method currently returns `umap: InverseTransform not yet implemented`; deferred while a pure-Go replacement for Python's Delaunay/QHull path is unresolved. |

## Reproducibility (seeded `RandSource`)

Use a fixed seed for deterministic runs:

```go
seed := uint64(42)
opts := umap.DefaultOptions()
opts.RandSource = umaprand.NewProduction(&seed)
model := umap.New(opts)
```

For repeatable independent runs, create a fresh `RandSource` from the same seed per run (do not reuse a previously advanced source).

## Usage

```go
package main

import (
	"github.com/nozzle/umap-go"
	umaprand "github.com/nozzle/umap-go/rand"
)

func seededOptions(seed uint64) umap.Options {
	opts := umap.DefaultOptions()
	opts.RandSource = umaprand.NewProduction(&seed)
	return opts
}
```

### 1) Unsupervised `Fit` and `FitTransform`

```go
var X [][]float64

modelA := umap.New(seededOptions(42))
if err := modelA.Fit(X); err != nil {
	panic(err)
}
embeddingA := modelA.Embedding()

modelB := umap.New(seededOptions(42))
embeddingB, err := modelB.FitTransform(X, nil)
if err != nil {
	panic(err)
}
_ = embeddingA
_ = embeddingB
```

### 2) Supervised `FitTransform`

```go
var X [][]float64
var y []float64 // class labels or continuous targets

model := umap.New(seededOptions(42))
embedding, err := model.FitTransform(X, y)
if err != nil {
	panic(err)
}
_ = embedding
```

### 3) Out-of-sample `Transform`

```go
var XTrain, XNew [][]float64

model := umap.New(seededOptions(42))
_, err := model.FitTransform(XTrain, nil)
if err != nil {
	panic(err)
}

newEmbedding, err := model.Transform(XNew)
if err != nil {
	panic(err)
}
_ = newEmbedding
```

## Python vs Go FitTransform benchmark

This repository includes a benchmark comparison for `FitTransform`:

- Python runner: `testdata/benchmark_fit_transform.py` (`umap-learn==0.5.11`)
- Go benchmark: `BenchmarkFitTransformCompare`

Run from repository root:

```bash
go test -run '^$' -bench BenchmarkFitTransformCompare -benchmem .
```

When Python tooling is available (`uv` or `python3` with `testdata` dependencies),
the Go benchmark reports extra metrics:

- `py_ns/op`: Python mean nanoseconds per operation
- `go_ns/op`: Go mean nanoseconds per operation
- `py/go`: Python-to-Go time ratio

If Python dependencies are unavailable, the benchmark still runs and reports Go-only metrics.

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](./LICENSE).

## Credits / Upstream Attribution

This library is a pure-Go port built for mathematical parity with:

- [`umap-learn` (v0.5.11)](https://github.com/lmcinnes/umap) — BSD 3-Clause
- [`pynndescent`](https://github.com/lmcinnes/pynndescent) — BSD 2-Clause

`umap-go` is an independent implementation and is not affiliated with or endorsed by the upstream projects.
