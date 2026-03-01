package umap

// layout.go implements the SGD-based layout optimization for UMAP.
//
// This is the core embedding optimization loop that moves points to minimize
// the cross-entropy between the high-dimensional fuzzy simplicial set and
// the low-dimensional representation.
//
// Corresponds to umap/layouts.py optimize_layout_euclidean().

import (
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/nozzle/umap-go/nn"
)

const (
	layoutModeSerial        = "serial"
	layoutModeDeterministic = "deterministic"
	layoutModeFast          = "fast"
)

// OptimizeLayoutConfig holds the configuration for layout optimization.
type OptimizeLayoutConfig struct {
	A                  float64 // UMAP a parameter
	B                  float64 // UMAP b parameter
	Gamma              float64 // repulsive force weight (default: 1.0)
	InitialAlpha       float64 // initial learning rate (default: 1.0)
	NegativeSampleRate float64 // negative samples per positive (default: 5.0)
	NEpochs            int     // number of optimization epochs
	MoveOther          bool    // update both head and tail embeddings
	NWorkers           int
	ParallelMode       string
}

// OptimizeLayoutEuclidean performs SGD optimization of the embedding layout.
//
// Parameters:
//
//	headEmbedding: initial embedding positions (modified in place)
//	tailEmbedding: reference embedding (same as headEmbedding for fit, different for transform)
//	head: source indices of edges
//	tail: target indices of edges
//	epochsPerSample: how many epochs between samples for each edge
//	rngState: per-sample Tausworthe PRNG states [nSamples][3]
//	cfg: optimization configuration
//
// Returns the optimized embedding (same slice as headEmbedding).
//
// Matches umap/layouts.py optimize_layout_euclidean().
func OptimizeLayoutEuclidean(
	headEmbedding [][]float64,
	tailEmbedding [][]float64,
	head []int,
	tail []int,
	epochsPerSample []float64,
	rngStates []nn.TauRandState,
	cfg OptimizeLayoutConfig,
) [][]float64 {
	nEdges := len(head)
	nVertices := len(headEmbedding)
	dim := len(headEmbedding[0])
	nEpochs := cfg.NEpochs

	// Defaults
	if cfg.Gamma == 0 {
		cfg.Gamma = 1.0
	}
	if cfg.InitialAlpha == 0 {
		cfg.InitialAlpha = 1.0
	}
	if cfg.NegativeSampleRate == 0 {
		cfg.NegativeSampleRate = 5.0
	}
	if cfg.NWorkers <= 0 {
		cfg.NWorkers = runtime.GOMAXPROCS(0)
	}
	if cfg.ParallelMode == "" {
		cfg.ParallelMode = "serial"
	}

	// Precompute epoch tracking
	epochOfNextSample := make([]float64, nEdges)
	copy(epochOfNextSample, epochsPerSample)

	epochOfNextNegativeSample := make([]float64, nEdges)
	for i, eps := range epochsPerSample {
		if eps > 0 {
			epochOfNextNegativeSample[i] = eps / cfg.NegativeSampleRate
		} else {
			epochOfNextNegativeSample[i] = -1
		}
	}

	nNegSamples := make([]int, nEdges)

	alpha := cfg.InitialAlpha
	const clipLimit = 4.0
	mode := resolveLayoutParallelMode(cfg.ParallelMode)
	useParallel := mode != layoutModeSerial && cfg.NWorkers > 1 && nEdges > 1
	vertexLocks := make([]sync.Mutex, nVertices)
	aliasEmbeddings := embeddingsAliased(headEmbedding, tailEmbedding)

	for epoch := range nEpochs {
		if useParallel {
			switch mode {
			case layoutModeDeterministic:
				optimizeSingleEpochParallelDeterministic(
					headEmbedding, tailEmbedding,
					head, tail,
					epochsPerSample,
					epochOfNextSample,
					epochOfNextNegativeSample,
					nNegSamples,
					rngStates,
					cfg.A, cfg.B, cfg.Gamma, alpha,
					cfg.NegativeSampleRate,
					nVertices, nEdges, dim,
					epoch, clipLimit,
					cfg.NWorkers, aliasEmbeddings,
				)
			case layoutModeFast:
				optimizeSingleEpochParallelFast(
					headEmbedding, tailEmbedding,
					head, tail,
					epochsPerSample,
					epochOfNextSample,
					epochOfNextNegativeSample,
					nNegSamples,
					rngStates,
					cfg.A, cfg.B, cfg.Gamma, alpha,
					cfg.NegativeSampleRate,
					nVertices, nEdges, dim,
					epoch, clipLimit,
					cfg.NWorkers, aliasEmbeddings, vertexLocks,
				)
			default:
				optimizeSingleEpoch(
					headEmbedding, tailEmbedding,
					head, tail,
					epochsPerSample,
					epochOfNextSample,
					epochOfNextNegativeSample,
					nNegSamples,
					rngStates,
					cfg.A, cfg.B, cfg.Gamma, alpha,
					cfg.NegativeSampleRate,
					nVertices, nEdges, dim,
					epoch, clipLimit,
				)
			}
		} else {
			optimizeSingleEpoch(
				headEmbedding, tailEmbedding,
				head, tail,
				epochsPerSample,
				epochOfNextSample,
				epochOfNextNegativeSample,
				nNegSamples,
				rngStates,
				cfg.A, cfg.B, cfg.Gamma, alpha,
				cfg.NegativeSampleRate,
				nVertices, nEdges, dim,
				epoch, clipLimit,
			)
		}

		// Linear learning rate decay
		alpha = cfg.InitialAlpha * (1.0 - float64(epoch+1)/float64(nEpochs))
	}

	return headEmbedding
}

// optimizeSingleEpoch runs one epoch of the SGD optimization.
// Matches _optimize_layout_euclidean_single_epoch in layouts.py.
func optimizeSingleEpoch(
	headEmbedding, tailEmbedding [][]float64,
	head, tail []int,
	epochsPerSample, epochOfNextSample, epochOfNextNegativeSample []float64,
	nNegSamples []int,
	rngStates []nn.TauRandState,
	a, b, gamma, alpha, negativeSampleRate float64,
	nVertices, nEdges, dim int,
	epoch int,
	clipLimit float64,
) {
	epochF := float64(epoch)

	for i := range nEdges {
		if epochOfNextSample[i] > epochF {
			continue
		}

		j := head[i]
		k := tail[i]

		current := headEmbedding[j]
		other := tailEmbedding[k]

		// Compute squared distance
		distSquared := 0.0
		for d := range dim {
			diff := current[d] - other[d]
			distSquared += diff * diff
		}

		// Attractive force
		if distSquared > 0 {
			gradCoeff := -2.0 * a * b * math.Pow(distSquared, b-1.0) /
				(1.0 + a*math.Pow(distSquared, b))

			for d := range dim {
				gradD := ClipValue(gradCoeff*(current[d]-other[d]), clipLimit)
				current[d] += gradD * alpha
			}
		}

		// Negative sampling
		nNegExpected := int((epochF - epochOfNextNegativeSample[i]) /
			(epochsPerSample[i] / negativeSampleRate))
		nNeg := nNegExpected - nNegSamples[i]
		nNegSamples[i] = nNegExpected

		for p := range nNeg {
			_ = p
			// Pick a random negative sample
			negIdx := nn.TauRandIntRange(&rngStates[i], nVertices)
			if negIdx == j {
				continue
			}

			negPoint := tailEmbedding[negIdx]

			distSquaredNeg := 0.0
			for d := range dim {
				diff := current[d] - negPoint[d]
				distSquaredNeg += diff * diff
			}

			if distSquaredNeg > 0 {
				gradCoeff := 2.0 * gamma * b /
					((0.001 + distSquaredNeg) * (1.0 + a*math.Pow(distSquaredNeg, b)))

				for d := range dim {
					gradD := ClipValue(gradCoeff*(current[d]-negPoint[d]), clipLimit)
					current[d] += gradD * alpha
				}
			}
		}

		epochOfNextSample[i] += epochsPerSample[i]
	}
}

func optimizeSingleEpochParallelDeterministic(
	headEmbedding, tailEmbedding [][]float64,
	head, tail []int,
	epochsPerSample, epochOfNextSample, epochOfNextNegativeSample []float64,
	nNegSamples []int,
	rngStates []nn.TauRandState,
	a, b, gamma, alpha, negativeSampleRate float64,
	nVertices, nEdges, dim int,
	epoch int,
	clipLimit float64,
	nWorkers int,
	aliasEmbeddings bool,
) {
	if nWorkers <= 1 || nEdges <= 1 {
		optimizeSingleEpoch(
			headEmbedding, tailEmbedding,
			head, tail,
			epochsPerSample, epochOfNextSample, epochOfNextNegativeSample,
			nNegSamples, rngStates,
			a, b, gamma, alpha, negativeSampleRate,
			nVertices, nEdges, dim, epoch, clipLimit,
		)
		return
	}
	if nWorkers > nEdges {
		nWorkers = nEdges
	}

	snapshotHead := cloneMatrix(headEmbedding)
	snapshotTail := tailEmbedding
	if aliasEmbeddings {
		snapshotTail = snapshotHead
	}

	type workerAccum struct {
		delta   []float64
		touched []int
	}
	accum := make([]workerAccum, nWorkers)
	epochF := float64(epoch)

	var wg sync.WaitGroup
	for w := range nWorkers {
		start := w * nEdges / nWorkers
		end := (w + 1) * nEdges / nWorkers
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(workerID, start, end int) {
			defer wg.Done()

			localDelta := make([]float64, nVertices*dim)
			touchedSeen := make([]bool, nVertices)
			touched := make([]int, 0, end-start)
			current := make([]float64, dim)

			for i := start; i < end; i++ {
				if epochOfNextSample[i] > epochF {
					continue
				}

				j := head[i]
				k := tail[i]
				if !touchedSeen[j] {
					touchedSeen[j] = true
					touched = append(touched, j)
				}
				copy(current, snapshotHead[j])
				other := snapshotTail[k]

				distSquared := 0.0
				for d := range dim {
					diff := current[d] - other[d]
					distSquared += diff * diff
				}

				if distSquared > 0 {
					gradCoeff := -2.0 * a * b * math.Pow(distSquared, b-1.0) /
						(1.0 + a*math.Pow(distSquared, b))

					for d := range dim {
						gradD := ClipValue(gradCoeff*(current[d]-other[d]), clipLimit)
						current[d] += gradD * alpha
					}
				}

				nNegExpected := int((epochF - epochOfNextNegativeSample[i]) /
					(epochsPerSample[i] / negativeSampleRate))
				nNeg := nNegExpected - nNegSamples[i]
				nNegSamples[i] = nNegExpected

				for range nNeg {
					negIdx := nn.TauRandIntRange(&rngStates[i], nVertices)
					if negIdx == j {
						continue
					}

					negPoint := snapshotTail[negIdx]
					distSquaredNeg := 0.0
					for d := range dim {
						diff := current[d] - negPoint[d]
						distSquaredNeg += diff * diff
					}

					if distSquaredNeg > 0 {
						gradCoeff := 2.0 * gamma * b /
							((0.001 + distSquaredNeg) * (1.0 + a*math.Pow(distSquaredNeg, b)))

						for d := range dim {
							gradD := ClipValue(gradCoeff*(current[d]-negPoint[d]), clipLimit)
							current[d] += gradD * alpha
						}
					}
				}

				base := j * dim
				for d := range dim {
					localDelta[base+d] += current[d] - snapshotHead[j][d]
				}

				epochOfNextSample[i] += epochsPerSample[i]
			}

			sort.Ints(touched)
			accum[workerID] = workerAccum{
				delta:   localDelta,
				touched: touched,
			}
		}(w, start, end)
	}
	wg.Wait()

	for w := range nWorkers {
		for _, idx := range accum[w].touched {
			base := idx * dim
			for d := range dim {
				headEmbedding[idx][d] += accum[w].delta[base+d]
			}
		}
	}
}

func optimizeSingleEpochParallelFast(
	headEmbedding, tailEmbedding [][]float64,
	head, tail []int,
	epochsPerSample, epochOfNextSample, epochOfNextNegativeSample []float64,
	nNegSamples []int,
	rngStates []nn.TauRandState,
	a, b, gamma, alpha, negativeSampleRate float64,
	nVertices, nEdges, dim int,
	epoch int,
	clipLimit float64,
	nWorkers int,
	aliasEmbeddings bool,
	vertexLocks []sync.Mutex,
) {
	if nWorkers <= 1 || nEdges <= 1 {
		optimizeSingleEpoch(
			headEmbedding, tailEmbedding,
			head, tail,
			epochsPerSample, epochOfNextSample, epochOfNextNegativeSample,
			nNegSamples, rngStates,
			a, b, gamma, alpha, negativeSampleRate,
			nVertices, nEdges, dim, epoch, clipLimit,
		)
		return
	}
	if nWorkers > nEdges {
		nWorkers = nEdges
	}

	epochF := float64(epoch)
	var wg sync.WaitGroup
	for w := range nWorkers {
		start := w * nEdges / nWorkers
		end := (w + 1) * nEdges / nWorkers
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			for i := start; i < end; i++ {
				if epochOfNextSample[i] > epochF {
					continue
				}

				j := head[i]
				k := tail[i]

				if aliasEmbeddings {
					lockVertexPair(vertexLocks, j, k)
				} else {
					vertexLocks[j].Lock()
				}

				current := headEmbedding[j]
				other := tailEmbedding[k]
				distSquared := 0.0
				for d := range dim {
					diff := current[d] - other[d]
					distSquared += diff * diff
				}

				if distSquared > 0 {
					gradCoeff := -2.0 * a * b * math.Pow(distSquared, b-1.0) /
						(1.0 + a*math.Pow(distSquared, b))
					for d := range dim {
						gradD := ClipValue(gradCoeff*(current[d]-other[d]), clipLimit)
						current[d] += gradD * alpha
					}
				}

				if aliasEmbeddings {
					unlockVertexPair(vertexLocks, j, k)
				} else {
					vertexLocks[j].Unlock()
				}

				nNegExpected := int((epochF - epochOfNextNegativeSample[i]) /
					(epochsPerSample[i] / negativeSampleRate))
				nNeg := nNegExpected - nNegSamples[i]
				nNegSamples[i] = nNegExpected

				for range nNeg {
					negIdx := nn.TauRandIntRange(&rngStates[i], nVertices)
					if negIdx == j {
						continue
					}

					if aliasEmbeddings {
						lockVertexPair(vertexLocks, j, negIdx)
					} else {
						vertexLocks[j].Lock()
					}

					current = headEmbedding[j]
					negPoint := tailEmbedding[negIdx]
					distSquaredNeg := 0.0
					for d := range dim {
						diff := current[d] - negPoint[d]
						distSquaredNeg += diff * diff
					}

					if distSquaredNeg > 0 {
						gradCoeff := 2.0 * gamma * b /
							((0.001 + distSquaredNeg) * (1.0 + a*math.Pow(distSquaredNeg, b)))
						for d := range dim {
							gradD := ClipValue(gradCoeff*(current[d]-negPoint[d]), clipLimit)
							current[d] += gradD * alpha
						}
					}

					if aliasEmbeddings {
						unlockVertexPair(vertexLocks, j, negIdx)
					} else {
						vertexLocks[j].Unlock()
					}
				}

				epochOfNextSample[i] += epochsPerSample[i]
			}
		}(start, end)
	}
	wg.Wait()
}

func resolveLayoutParallelMode(mode string) string {
	switch mode {
	case "", "serial":
		return layoutModeSerial
	case "auto", "deterministic":
		return layoutModeDeterministic
	case "parallel", "fast":
		return layoutModeFast
	default:
		return layoutModeSerial
	}
}

func embeddingsAliased(a, b [][]float64) bool {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return false
	}
	if len(a[0]) == 0 || len(b[0]) == 0 {
		return false
	}
	return &a[0][0] == &b[0][0]
}

func cloneMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = make([]float64, len(m[i]))
		copy(out[i], m[i])
	}
	return out
}

func lockVertexPair(locks []sync.Mutex, a, b int) {
	if a == b {
		locks[a].Lock()
		return
	}
	if a < b {
		locks[a].Lock()
		locks[b].Lock()
		return
	}
	locks[b].Lock()
	locks[a].Lock()
}

func unlockVertexPair(locks []sync.Mutex, a, b int) {
	if a == b {
		locks[a].Unlock()
		return
	}
	if a < b {
		locks[b].Unlock()
		locks[a].Unlock()
		return
	}
	locks[a].Unlock()
	locks[b].Unlock()
}

// OptimizeLayoutGeneric performs SGD with a custom output distance metric.
// Used for inverse_transform or non-Euclidean output spaces.
//
// TODO: Implement this for inverse_transform support.
func OptimizeLayoutGeneric(
	headEmbedding [][]float64,
	tailEmbedding [][]float64,
	head []int,
	tail []int,
	epochsPerSample []float64,
	rngStates []nn.TauRandState,
	cfg OptimizeLayoutConfig,
	outputMetricGrad func(x, y []float64) (float64, []float64),
) [][]float64 {
	// TODO: Implement generic layout optimization with custom output metric
	// For now, fall back to Euclidean optimization
	return OptimizeLayoutEuclidean(
		headEmbedding, tailEmbedding,
		head, tail,
		epochsPerSample, rngStates,
		cfg,
	)
}
