package nn

import (
	"github.com/nozzle/umap-go/distance"
	umaprand "github.com/nozzle/umap-go/rand"
	"math"
	"runtime"
	"sync"
)

// NNDescentConfig holds the configuration for NN-Descent.
type NNDescentConfig struct {
	K             int             // number of nearest neighbors
	NTrees        int             // number of RP-trees (default: min(32, 5 + int(round(n^0.25))))
	MaxCandidates int             // max candidates per iteration (default: min(60, K))
	NIters        int             // max iterations (default: max(5, int(round(log2(n)))))
	Delta         float64         // early stopping threshold (default: 0.001)
	LeafSize      int             // RP-tree leaf size (default: max(10, K))
	Angular       bool            // use angular RP-trees (for cosine-like metrics)
	LowMemory     bool            // use low-memory mode (default: true)
	Rng           umaprand.Source // random source for RP-tree construction
	Verbose       bool
	NWorkers      int
	ParallelMode  string
}

// NNDescentResult holds the output of NN-Descent.
type NNDescentResult struct {
	Indices   [][]int     // [n][k] neighbor indices
	Distances [][]float64 // [n][k] neighbor distances
	Forest    RPForest    // the RP-forest used for initialization
}

// NNDescent performs approximate nearest neighbor search using NN-Descent.
// This matches the algorithm in pynndescent (v0.5.x).
//
// Algorithm:
// 1. Build RP-forest for initial candidate generation
// 2. Initialize heap from forest leaf co-occurrence
// 3. Iterate: for each point, compare its neighbors' neighbors as candidates
// 4. Stop when fewer than delta*n*k updates per iteration
func NNDescent(data [][]float64, distFunc distance.Func, cfg NNDescentConfig) *NNDescentResult {
	n := len(data)

	// Set defaults
	if cfg.K <= 0 {
		cfg.K = 15
	}
	if cfg.MaxCandidates <= 0 {
		cfg.MaxCandidates = min(cfg.K, 60)
	}
	if cfg.NIters <= 0 {
		cfg.NIters = maxInt(5, int(math.Round(math.Log2(float64(n)))))
	}
	if cfg.Delta <= 0 {
		cfg.Delta = 0.001
	}
	if cfg.LeafSize <= 0 {
		cfg.LeafSize = maxInt(60, minInt(256, 5*cfg.K))
	}
	if cfg.NTrees <= 0 {
		cfg.NTrees = defaultNTrees(n)
	}
	if cfg.NWorkers <= 0 {
		cfg.NWorkers = runtime.GOMAXPROCS(0)
	}
	if cfg.ParallelMode == "" {
		cfg.ParallelMode = "serial"
	}

	// Step 1: Initialize RNG states
	rngState := makeTauRandState(cfg.Rng)
	searchRngState := makeTauRandState(cfg.Rng)
	for range 10 {
		_ = TauRandInt(&searchRngState)
	}

	// Step 2: Build RP-forest
	var forest RPForest
	if cfg.Rng != nil {
		forest = MakeForest(data, cfg.NTrees, cfg.LeafSize, cfg.Rng, cfg.Angular)
	}

	// Step 3: Initialize heap
	var heap *Heap
	if forest != nil {
		heap = InitFromForest(forest, data, cfg.K, distFunc)
	} else {
		heap = NewHeap(n, cfg.K)
	}

	// Fill any remaining spots with random candidates
	heap = InitFromRandom(heap, n, cfg.K, distFunc, data, &rngState)

	// Step 4: NN-Descent iterations
	threshold := cfg.Delta * float64(cfg.K) * float64(n)

	for iter := range cfg.NIters {
		newCands, oldCands := heap.BuildCandidates(cfg.MaxCandidates, &searchRngState)

		updates := 0
		updates += processNewCandidates(heap, newCands, data, distFunc, cfg.NWorkers, cfg.ParallelMode)
		updates += processNewOldCandidates(heap, newCands, oldCands, data, distFunc, cfg.NWorkers, cfg.ParallelMode)

		if cfg.Verbose {
			_ = iter // suppress unused warning in non-verbose mode
		}

		if float64(updates) < threshold {
			break
		}
	}

	// Step 4: Sort results
	heap.Deheap()

	return &NNDescentResult{
		Indices:   heap.Indices,
		Distances: heap.Distances,
		Forest:    forest,
	}
}

// processNewCandidates processes all pairs of new candidates.
// For each point, compare all pairs of its new candidate neighbors.
func processNewCandidates(
	heap *Heap,
	newCands *Heap,
	data [][]float64,
	distFunc distance.Func,
	nWorkers int,
	parallelMode string,
) int {
	if shouldParallelCandidateEval(parallelMode, nWorkers, newCands.N) {
		return processNewCandidatesParallel(heap, newCands, data, distFunc, nWorkers)
	}

	updates := 0

	for i := range newCands.N {
		for j := 0; j < newCands.K; j++ {
			p := newCands.Indices[i][j]
			if p < 0 {
				continue
			}
			for k := j + 1; k < newCands.K; k++ {
				q := newCands.Indices[i][k]
				if q < 0 {
					continue
				}

				d := distFunc(data[p], data[q])
				if heap.Push(p, q, d, true) {
					updates++
				}
				if heap.Push(q, p, d, true) {
					updates++
				}
			}
		}
	}

	return updates
}

// processNewOldCandidates processes pairs of (new, old) candidates.
func processNewOldCandidates(
	heap *Heap,
	newCands, oldCands *Heap,
	data [][]float64,
	distFunc distance.Func,
	nWorkers int,
	parallelMode string,
) int {
	if shouldParallelCandidateEval(parallelMode, nWorkers, newCands.N) {
		return processNewOldCandidatesParallel(heap, newCands, oldCands, data, distFunc, nWorkers)
	}

	updates := 0

	for i := range newCands.N {
		for j := 0; j < newCands.K; j++ {
			p := newCands.Indices[i][j]
			if p < 0 {
				continue
			}
			for k := 0; k < oldCands.K; k++ {
				q := oldCands.Indices[i][k]
				if q < 0 || p == q {
					continue
				}

				d := distFunc(data[p], data[q])
				if heap.Push(p, q, d, true) {
					updates++
				}
				if heap.Push(q, p, d, true) {
					updates++
				}
			}
		}
	}

	return updates
}

type candidatePair struct {
	p int
	q int
	d float64
}

func shouldParallelCandidateEval(mode string, nWorkers, nRows int) bool {
	if nWorkers <= 1 || nRows <= 1 {
		return false
	}
	switch mode {
	case "parallel", "auto":
		return true
	default:
		return false
	}
}

func processNewCandidatesParallel(
	heap *Heap,
	newCands *Heap,
	data [][]float64,
	distFunc distance.Func,
	nWorkers int,
) int {
	type rowResult struct {
		row   int
		pairs []candidatePair
	}

	rowCount := newCands.N
	workerCount := minInt(nWorkers, rowCount)
	jobs := make(chan int, workerCount)
	results := make(chan rowResult, workerCount)

	var wg sync.WaitGroup
	for range workerCount {
		wg.Go(func() {
			for i := range jobs {
				pairs := make([]candidatePair, 0, newCands.K)
				for j := 0; j < newCands.K; j++ {
					p := newCands.Indices[i][j]
					if p < 0 {
						continue
					}
					for k := j + 1; k < newCands.K; k++ {
						q := newCands.Indices[i][k]
						if q < 0 {
							continue
						}
						pairs = append(pairs, candidatePair{
							p: p,
							q: q,
							d: distFunc(data[p], data[q]),
						})
					}
				}
				results <- rowResult{row: i, pairs: pairs}
			}
		})
	}

	go func() {
		for i := range rowCount {
			jobs <- i
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	updates := 0
	nextRow := 0
	pending := make(map[int][]candidatePair, workerCount)
	for res := range results {
		pending[res.row] = res.pairs
		for {
			pairs, ok := pending[nextRow]
			if !ok {
				break
			}
			updates += applyCandidatePairs(heap, pairs)
			delete(pending, nextRow)
			nextRow++
		}
	}

	return updates
}

func processNewOldCandidatesParallel(
	heap *Heap,
	newCands, oldCands *Heap,
	data [][]float64,
	distFunc distance.Func,
	nWorkers int,
) int {
	type rowResult struct {
		row   int
		pairs []candidatePair
	}

	rowCount := newCands.N
	workerCount := minInt(nWorkers, rowCount)
	jobs := make(chan int, workerCount)
	results := make(chan rowResult, workerCount)

	var wg sync.WaitGroup
	for range workerCount {
		wg.Go(func() {
			for i := range jobs {
				pairs := make([]candidatePair, 0, newCands.K)
				for j := 0; j < newCands.K; j++ {
					p := newCands.Indices[i][j]
					if p < 0 {
						continue
					}
					for k := 0; k < oldCands.K; k++ {
						q := oldCands.Indices[i][k]
						if q < 0 || p == q {
							continue
						}
						pairs = append(pairs, candidatePair{
							p: p,
							q: q,
							d: distFunc(data[p], data[q]),
						})
					}
				}
				results <- rowResult{row: i, pairs: pairs}
			}
		})
	}

	go func() {
		for i := range rowCount {
			jobs <- i
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	updates := 0
	nextRow := 0
	pending := make(map[int][]candidatePair, workerCount)
	for res := range results {
		pending[res.row] = res.pairs
		for {
			pairs, ok := pending[nextRow]
			if !ok {
				break
			}
			updates += applyCandidatePairs(heap, pairs)
			delete(pending, nextRow)
			nextRow++
		}
	}

	return updates
}

func applyCandidatePairs(heap *Heap, pairs []candidatePair) int {
	updates := 0
	for _, pair := range pairs {
		if heap.Push(pair.p, pair.q, pair.d, true) {
			updates++
		}
		if heap.Push(pair.q, pair.p, pair.d, true) {
			updates++
		}
	}
	return updates
}

// NearestNeighbors computes nearest neighbors, choosing brute-force or
// NN-Descent based on dataset size. This matches the dispatch logic in
// umap_.py nearest_neighbors().
//
// For n < 4096: brute-force pairwise distances.
// For n >= 4096: NN-Descent.
func NearestNeighbors(data [][]float64, k int, distFunc distance.Func, rng umaprand.Source, angular bool) *SearchIndex {
	return NearestNeighborsWithConfig(data, k, distFunc, rng, angular, NNDescentConfig{})
}

// NearestNeighborsWithConfig computes nearest neighbors with NN-Descent controls.
func NearestNeighborsWithConfig(
	data [][]float64,
	k int,
	distFunc distance.Func,
	rng umaprand.Source,
	angular bool,
	cfg NNDescentConfig,
) *SearchIndex {
	n := len(data)

	if n < 4096 {
		indices, distances := BruteForceKNN(data, k, distFunc)
		idx := NewSearchIndex(data, indices, nil, distFunc, rng)
		// Brute force does not populate forest, but we can store distances inside SearchIndex if we want,
		// or just rely on a new field. We'll store distances in SearchIndex just in case.
		idx.Distances = distances
		idx.isBruteForce = true
		return idx
	}

	cfg.K = k
	cfg.Rng = rng
	cfg.Angular = angular
	result := NNDescent(data, distFunc, cfg)

	// Create hub tree for search index
	searchRngState := makeTauRandState(rng)
	searchLeafSize := max(cfg.K, 30)
	hubTree := MakeHubTree(data, result.Indices, searchLeafSize, &searchRngState, angular, 200)
	searchForest := RPForest{hubTree}

	idx := NewSearchIndex(data, result.Indices, searchForest, distFunc, rng)
	idx.Distances = result.Distances
	return idx
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func defaultNTrees(n int) int {
	// Matches UMAP default: min(64, 5 + int(round((n^0.5)/20.0)))
	import_math := 5 + int(0.5+sqrt(float64(n))/20.0)
	if import_math > 64 {
		return 64
	}
	return import_math
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	z := x / 2
	for range 100 {
		z2 := (z + x/z) / 2
		if z2 == z {
			break
		}
		z = z2
	}
	return z
}

func makeTauRandState(rng umaprand.Source) TauRandState {
	if rng == nil {
		return TauRandState{42, 13, 7}
	}
	var state TauRandState
	for i := range 3 {
		// Python randint(-2147483647, 2147483646) draws from a range of 4294967293
		state[i] = int64(rng.Intn(4294967293) - 2147483647)
	}
	return state
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
