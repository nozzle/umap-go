package umap

// options.go defines the UMAP configuration options.
//
// All options have sensible defaults matching Python UMAP v0.5.11.

import (
	"fmt"
	"math"
	"runtime"

	"github.com/nozzle/umap-go/distance"
	umaprand "github.com/nozzle/umap-go/rand"
)

// Options configures the UMAP algorithm.
type Options struct {
	// NNeighbors is the number of nearest neighbors to use for graph construction.
	// Larger values capture more global structure at the cost of local detail.
	// Default: 15.
	NNeighbors int

	// NComponents is the dimensionality of the output embedding.
	// Default: 2.
	NComponents int

	// MinDist controls how tightly points are packed together.
	// Smaller values produce more clustered embeddings.
	// Default: 0.1.
	MinDist float64

	// Spread determines the scale of the embedding.
	// Together with MinDist, controls the membership function.
	// Default: 1.0.
	Spread float64

	// Metric is the distance metric to use on the input data.
	// Default: "euclidean".
	Metric string

	// MetricKwds are additional parameters for parameterized metrics
	// (e.g., "p" for Minkowski, "sigma" for StandardisedEuclidean).
	MetricKwds map[string]any

	// NEpochs is the number of SGD optimization epochs.
	// If 0, automatically chosen based on dataset size.
	NEpochs int

	// InitMethod controls the initial embedding.
	// "spectral" (default), "random", or a custom [][]float64.
	InitMethod string

	// CustomInit provides a pre-computed initial embedding.
	// Only used if InitMethod == "custom".
	CustomInit [][]float64

	// LocalConnectivity is the number of nearest neighbors that should
	// be assumed to be connected at a local level.
	// Default: 1.0.
	LocalConnectivity float64

	// SetOpMixRatio controls the blend between fuzzy union and intersection
	// for the symmetrization of the kNN graph.
	// 1.0 = pure union (default), 0.0 = pure intersection.
	SetOpMixRatio float64

	// DisconnectionDistance removes edges with distances greater than or equal to this value.
	// Default: +inf (no disconnection).
	DisconnectionDistance float64

	// NegativeSampleRate controls the number of negative samples per
	// positive sample in SGD optimization.
	// Default: 5.
	NegativeSampleRate float64

	// RepulsionStrength controls the weight of the repulsive force.
	// Default: 1.0.
	RepulsionStrength float64

	// LearningRate is the initial SGD learning rate.
	// Default: 1.0.
	LearningRate float64

	// RandSource provides the random number generator.
	// Default: Production (wrapping math/rand/v2 with seed 42).
	RandSource umaprand.Source

	// TargetNNeighbors is the number of nearest neighbors for the target
	// (y) space in supervised mode.
	// Default: NNeighbors.
	TargetNNeighbors int

	// TargetMetric is the distance metric for the target space.
	// Default: "categorical" for discrete labels, "euclidean" for continuous.
	TargetMetric string

	// TargetWeight controls the balance between data topology and target
	// topology in supervised mode. 0.0 = data only, 1.0 = target only.
	// Default: 0.5.
	TargetWeight float64

	// Verbose controls whether progress information is printed.
	Verbose bool

	// NWorkers controls the number of worker goroutines used by parallel-capable stages.
	// If 0, runtime.GOMAXPROCS(0) is used.
	// Default: runtime.GOMAXPROCS(0).
	NWorkers int

	// ParallelMode controls how parallel-capable stages execute.
	// "auto" (default), "serial", or "parallel".
	ParallelMode string

	// TODO: DensMAP parameters (dens_lambda, dens_frac, dens_var_shift)
	// TODO: Disconnection distance
	// TODO: Output metric / output metric kwds
	// TODO: Precomputed kNN
}

// DefaultOptions returns Options with all defaults set.
func DefaultOptions() Options {
	return Options{
		NNeighbors:         15,
		NComponents:        2,
		MinDist:            0.1,
		Spread:             1.0,
		Metric:             "euclidean",
		NEpochs:            0,
		InitMethod:         "spectral",
		LocalConnectivity:  1.0,
		SetOpMixRatio:      1.0,
		NegativeSampleRate: 5,
		RepulsionStrength:  1.0,
		LearningRate:       1.0,
		RandSource:         umaprand.NewProduction(new(uint64(42))),
		TargetNNeighbors:   15,
		TargetMetric:       "categorical",
		TargetWeight:       0.5,
		NWorkers:           runtime.GOMAXPROCS(0),
		ParallelMode:       "auto",
	}
}

// applyDefaults fills in zero-value fields with defaults.
func (o *Options) applyDefaults() {
	if o.NNeighbors == 0 {
		o.NNeighbors = 15
	}
	if o.NComponents == 0 {
		o.NComponents = 2
	}
	if o.MinDist == 0 {
		o.MinDist = 0.1
	}
	if o.Spread == 0 {
		o.Spread = 1.0
	}
	if o.Metric == "" {
		o.Metric = "euclidean"
	}
	if o.InitMethod == "" {
		o.InitMethod = "spectral"
	}
	if o.LocalConnectivity == 0 {
		o.LocalConnectivity = 1.0
	}
	if o.SetOpMixRatio < 0 {
		o.SetOpMixRatio = 1.0
	}
	if o.NegativeSampleRate == 0 {
		o.NegativeSampleRate = 5
	}
	if o.RepulsionStrength == 0 {
		o.RepulsionStrength = 1.0
	}
	if o.LearningRate == 0 {
		o.LearningRate = 1.0
	}
	if o.RandSource == nil {
		o.RandSource = umaprand.NewProduction(new(uint64(42)))
	}
	if o.TargetNNeighbors == 0 {
		o.TargetNNeighbors = o.NNeighbors
	}
	if o.TargetMetric == "" {
		o.TargetMetric = "categorical"
	}
	if o.TargetWeight < 0 {
		o.TargetWeight = 0.5
	}
	if o.NWorkers == 0 {
		o.NWorkers = runtime.GOMAXPROCS(0)
	}
	if o.ParallelMode == "" {
		o.ParallelMode = "auto"
	}
	if o.DisconnectionDistance == 0 {
		o.DisconnectionDistance = 1e308
	}
}

func (o Options) validate() error {
	if o.NNeighbors < 2 {
		return fmt.Errorf("umap: invalid NNeighbors %d: must be >= 2", o.NNeighbors)
	}
	if o.NComponents < 1 {
		return fmt.Errorf("umap: invalid NComponents %d: must be >= 1", o.NComponents)
	}
	if o.NEpochs < 0 {
		return fmt.Errorf("umap: invalid NEpochs %d: must be >= 0", o.NEpochs)
	}
	if o.MinDist < 0 {
		return fmt.Errorf("umap: invalid MinDist %g: must be >= 0", o.MinDist)
	}
	if o.Spread <= 0 {
		return fmt.Errorf("umap: invalid Spread %g: must be > 0", o.Spread)
	}
	if o.MinDist > o.Spread {
		return fmt.Errorf("umap: invalid MinDist %g: must be <= Spread %g", o.MinDist, o.Spread)
	}
	if !isSupportedMetric(o.Metric) {
		return fmt.Errorf("umap: invalid Metric %q: unsupported metric", o.Metric)
	}
	switch o.InitMethod {
	case "spectral", "random", "custom":
	default:
		return fmt.Errorf("umap: invalid InitMethod %q: must be one of spectral, random, custom", o.InitMethod)
	}
	if o.LocalConnectivity < 0 {
		return fmt.Errorf("umap: invalid LocalConnectivity %g: must be >= 0", o.LocalConnectivity)
	}
	if o.SetOpMixRatio < 0 || o.SetOpMixRatio > 1 {
		return fmt.Errorf("umap: invalid SetOpMixRatio %g: must be in [0, 1]", o.SetOpMixRatio)
	}
	if o.NegativeSampleRate <= 0 {
		return fmt.Errorf("umap: invalid NegativeSampleRate %g: must be > 0", o.NegativeSampleRate)
	}
	if o.RepulsionStrength <= 0 {
		return fmt.Errorf("umap: invalid RepulsionStrength %g: must be > 0", o.RepulsionStrength)
	}
	if o.LearningRate <= 0 {
		return fmt.Errorf("umap: invalid LearningRate %g: must be > 0", o.LearningRate)
	}
	if o.NWorkers < 1 {
		return fmt.Errorf("umap: invalid NWorkers %d: must be >= 1", o.NWorkers)
	}
	switch o.ParallelMode {
	case "serial", "parallel", "auto":
	default:
		return fmt.Errorf("umap: invalid ParallelMode %q: must be one of auto, serial, parallel", o.ParallelMode)
	}
	if o.TargetNNeighbors < 2 {
		return fmt.Errorf("umap: invalid TargetNNeighbors %d: must be >= 2", o.TargetNNeighbors)
	}
	if !isSupportedMetric(o.TargetMetric) {
		return fmt.Errorf("umap: invalid TargetMetric %q: unsupported metric", o.TargetMetric)
	}
	if o.TargetWeight < 0 || o.TargetWeight > 1 {
		return fmt.Errorf("umap: invalid TargetWeight %g: must be in [0, 1]", o.TargetWeight)
	}
	if math.IsNaN(o.DisconnectionDistance) || o.DisconnectionDistance < 0 {
		return fmt.Errorf("umap: invalid DisconnectionDistance %g: must be >= 0", o.DisconnectionDistance)
	}
	return nil
}

func isSupportedMetric(name string) bool {
	return name == "categorical" || distance.Named(name) != nil || distance.NamedWithParam(name) != nil
}
