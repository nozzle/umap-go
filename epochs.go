package umap

// epochs.go implements make_epochs_per_sample.
//
// Given the weights of edges in the fuzzy simplicial set graph, this
// determines how many epochs should pass between samples of each edge.
// Edges with higher weights are sampled more frequently.
//
// Corresponds to umap_.py make_epochs_per_sample().

// MakeEpochsPerSample computes the number of epochs between samples
// for each edge weight.
//
// Given weights and n_epochs:
//
//	n_samples = n_epochs * (weight / max_weight)
//	epochs_per_sample = n_epochs / n_samples
//
// Which simplifies to: epochs_per_sample = max_weight / weight
//
// Weights that are zero or negative get -1 (never sampled).
//
// Matches umap_.py make_epochs_per_sample() exactly.
func MakeEpochsPerSample(weights []float64, nEpochs int) []float64 {
	// Find max weight
	maxWeight := 0.0
	for _, w := range weights {
		if w > maxWeight {
			maxWeight = w
		}
	}

	result := make([]float64, len(weights))

	if maxWeight == 0 {
		for i := range result {
			result[i] = -1
		}
		return result
	}

	for i, w := range weights {
		nSamples := float64(nEpochs) * (w / maxWeight)
		if nSamples > 0 {
			result[i] = float64(nEpochs) / nSamples
		} else {
			result[i] = -1
		}
	}

	return result
}

// SelectNEpochs chooses the number of optimization epochs based on
// dataset size if the user hasn't specified one.
// Matches umap_.py default n_epochs selection.
func SelectNEpochs(nSamples int) int {
	if nSamples <= 10000 {
		return 500
	}
	return 200
}

// EpochsOfNextSample precomputes the epoch at which each edge should
// first be sampled. Used to drive the SGD loop.
func EpochsOfNextSample(epochsPerSample []float64) []float64 {
	result := make([]float64, len(epochsPerSample))
	copy(result, epochsPerSample)
	return result
}

// EpochsOfNextNegativeSample precomputes the epoch at which each edge
// should next receive a negative sample.
func EpochsOfNextNegativeSample(epochsPerSample []float64, negativeSampleRate float64) []float64 {
	result := make([]float64, len(epochsPerSample))
	for i, eps := range epochsPerSample {
		if eps > 0 {
			result[i] = eps / negativeSampleRate
		} else {
			result[i] = -1
		}
	}
	return result
}

// ClipValue clamps a value to [-clipLimit, clipLimit].
// Used in SGD gradient updates. Matches UMAP's clip().
func ClipValue(x, clipLimit float64) float64 {
	if x > clipLimit {
		return clipLimit
	}
	if x < -clipLimit {
		return -clipLimit
	}
	return x
}

// Rdist computes the squared Euclidean distance between two vectors.
// Used in the SGD inner loop. Matches UMAP's rdist().
func Rdist(x, y []float64) float64 {
	var sum float64
	for i := range x {
		d := x[i] - y[i]
		sum += d * d
	}
	return sum
}
