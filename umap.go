package umap

// umap.go implements the main UMAP API: UMAP struct with Fit, FitTransform,
// Transform, and InverseTransform methods.
//
// Corresponds to umap_.py UMAP class.

import (
	"fmt"
	"math"

	"github.com/nozzle/umap-go/nn"
	"github.com/nozzle/umap-go/sparse"
)

// UMAP is the main UMAP dimensionality reduction model.
type UMAP struct {
	opts Options

	// Fitted state
	graph     *sparse.CSR // the fuzzy simplicial set graph
	embedding [][]float64 // the fitted embedding
	a, b      float64     // curve-fitting parameters

	// Data retained for transform
	rawData    [][]float64
	knnIndices [][]int
	knnDists   [][]float64

	fitted bool
}

// New creates a new UMAP model with the given options.
func New(opts Options) *UMAP {
	opts.applyDefaults()
	return &UMAP{opts: opts}
}

// Fit fits the UMAP model to the data X without returning the embedding.
// X is a matrix of shape (n_samples, n_features) stored as [][]float64.
func (u *UMAP) Fit(X [][]float64) error {
	_, err := u.FitTransform(X, nil)
	return err
}

// FitTransform fits the UMAP model and returns the embedding.
// X: input data (n_samples x n_features)
// y: optional target labels for supervised mode (nil for unsupervised)
func (u *UMAP) FitTransform(X [][]float64, y []float64) ([][]float64, error) {
	if len(X) == 0 {
		return nil, fmt.Errorf("umap: empty input data")
	}
	if len(X[0]) == 0 {
		return nil, fmt.Errorf("umap: zero-dimensional input data")
	}

	n := len(X)
	u.rawData = X

	// Step 1: Find a, b parameters
	u.a, u.b = FindABParams(u.opts.Spread, u.opts.MinDist)

	// Step 2: Construct fuzzy simplicial set
	result := FuzzySimplicialSet(
		X,
		u.opts.NNeighbors,
		u.opts.RandSource,
		u.opts.Metric,
		u.opts.MetricKwds,
		u.opts.LocalConnectivity,
		u.opts.SetOpMixRatio,
	)
	u.graph = result.Graph

	// Step 3: Supervised mode — intersect with target graph
	if y != nil && len(y) == n {
		u.applySupervisedGraph(y)
	}

	// Step 4: Determine n_epochs
	nEpochs := u.opts.NEpochs
	if nEpochs == 0 {
		nEpochs = SelectNEpochs(n)
	}

	// Step 5: Compute initial embedding
	embedding := u.computeInitialEmbedding(n)

	// Step 6: Prepare SGD data
	graphCOO := graphToEdges(u.graph)

	epochsPerSample := MakeEpochsPerSample(graphCOO.weights, nEpochs)

	// Filter out edges that will never be sampled
	var head, tail []int
	var activeEPS []float64
	for i, eps := range epochsPerSample {
		if eps > 0 {
			head = append(head, graphCOO.rows[i])
			tail = append(tail, graphCOO.cols[i])
			activeEPS = append(activeEPS, eps)
		}
	}

	// Initialize per-edge RNG states
	rngStates := make([]nn.TauRandState, len(head))
	for i := range rngStates {
		s := u.opts.RandSource.Intn(math.MaxInt32)
		rngStates[i] = nn.TauRandState{int64(s), int64(s + 1), int64(s + 2)}
	}

	// Step 7: Optimize layout
	cfg := OptimizeLayoutConfig{
		A:                  u.a,
		B:                  u.b,
		Gamma:              u.opts.RepulsionStrength,
		InitialAlpha:       u.opts.LearningRate,
		NegativeSampleRate: u.opts.NegativeSampleRate,
		NEpochs:            nEpochs,
	}

	embedding = OptimizeLayoutEuclidean(
		embedding, embedding,
		head, tail,
		activeEPS,
		rngStates,
		cfg,
	)

	u.embedding = embedding
	u.fitted = true

	return embedding, nil
}

// Transform projects new data into the existing embedding space.
// The model must be fitted first.
//
// TODO: Full implementation with kNN search into training data.
func (u *UMAP) Transform(XNew [][]float64) ([][]float64, error) {
	if !u.fitted {
		return nil, fmt.Errorf("umap: model not fitted; call Fit or FitTransform first")
	}
	if len(XNew) == 0 {
		return nil, fmt.Errorf("umap: empty input data")
	}

	// TODO: Implement full transform:
	// 1. Find kNN from XNew into training data
	// 2. Build membership strengths
	// 3. Initialize from weighted average of training embedding
	// 4. SGD optimize with training embedding frozen
	return nil, fmt.Errorf("umap: Transform not yet implemented")
}

// InverseTransform maps points from the embedding space back to data space.
//
// TODO: Implement inverse transform.
func (u *UMAP) InverseTransform(XEmbedded [][]float64) ([][]float64, error) {
	if !u.fitted {
		return nil, fmt.Errorf("umap: model not fitted; call Fit or FitTransform first")
	}
	// TODO: Implement inverse transform:
	// 1. Find kNN in embedding space
	// 2. Weighted combination of training data
	// 3. SGD optimization with gradient-equipped output metric
	return nil, fmt.Errorf("umap: InverseTransform not yet implemented")
}

// Embedding returns the fitted embedding. Nil if not fitted.
func (u *UMAP) Embedding() [][]float64 {
	return u.embedding
}

// Graph returns the fitted fuzzy simplicial set graph. Nil if not fitted.
func (u *UMAP) Graph() *sparse.CSR {
	return u.graph
}

// A returns the fitted 'a' parameter.
func (u *UMAP) A() float64 { return u.a }

// B returns the fitted 'b' parameter.
func (u *UMAP) B() float64 { return u.b }

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// computeInitialEmbedding generates the initial embedding.
func (u *UMAP) computeInitialEmbedding(n int) [][]float64 {
	nComp := u.opts.NComponents

	switch u.opts.InitMethod {
	case "spectral":
		init := SpectralLayout(u.graph, nComp)
		if init != nil {
			return init
		}
		// Fallback to random
		return u.randomInitEmbedding(n, nComp)

	case "random":
		return u.randomInitEmbedding(n, nComp)

	case "custom":
		if u.opts.CustomInit != nil {
			return u.opts.CustomInit
		}
		return u.randomInitEmbedding(n, nComp)

	default:
		return u.randomInitEmbedding(n, nComp)
	}
}

// randomInitEmbedding generates a random initial embedding.
func (u *UMAP) randomInitEmbedding(n, nComp int) [][]float64 {
	embedding := make([][]float64, n)
	for i := range n {
		embedding[i] = make([]float64, nComp)
		for d := range nComp {
			embedding[i][d] = u.opts.RandSource.UniformFloat64(-10, 10)
		}
	}
	return embedding
}

// applySupervisedGraph constructs and intersects a target topology.
func (u *UMAP) applySupervisedGraph(y []float64) {
	// Build target graph based on label equivalence
	n := len(y)
	coo := sparse.NewCOO(n, n)

	for i := range n {
		for j := range n {
			if i != j {
				if y[i] == y[j] {
					coo.Set(i, j, 1.0)
				} else {
					coo.Set(i, j, 0.1) // small weight for different labels
				}
			}
		}
	}

	targetGraph := coo.ToCSR()

	// Intersect data graph with target graph
	u.graph = sparse.GeneralSimplicialSetIntersection(u.graph, targetGraph, u.opts.TargetWeight)
	u.graph = sparse.ResetLocalConnectivity(u.graph)
}

// edgeList is a COO-like list of edges.
type edgeList struct {
	rows    []int
	cols    []int
	weights []float64
}

// graphToEdges converts a CSR graph to an edge list.
func graphToEdges(graph *sparse.CSR) *edgeList {
	var rows, cols []int
	var weights []float64

	for i := range graph.Rows {
		for idx := graph.Indptr[i]; idx < graph.Indptr[i+1]; idx++ {
			rows = append(rows, i)
			cols = append(cols, graph.Indices[idx])
			weights = append(weights, graph.Data[idx])
		}
	}

	return &edgeList{rows: rows, cols: cols, weights: weights}
}
