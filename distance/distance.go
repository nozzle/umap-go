// Package distance provides distance metric implementations for UMAP.
//
// All metrics match the implementations in umap/distances.py (v0.5.11).
// Each function operates on dense float64 vectors unless noted otherwise.
//
// Sparse variants operate on CSR row slices (sorted index + data arrays).
package distance

// Func computes the distance between two dense float64 vectors.
type Func func(x, y []float64) float64

// FuncWithParam computes distance with additional parameters.
type FuncWithParam func(x, y []float64, params map[string]interface{}) float64

// GradFunc computes distance and its gradient with respect to x.
// Returns (distance, gradient).
type GradFunc func(x, y []float64) (float64, []float64)

// GradFuncWithParam computes distance and gradient with additional parameters.
type GradFuncWithParam func(x, y []float64, params map[string]interface{}) (float64, []float64)

// SparseFunc computes distance between two sparse vectors represented as
// sorted index arrays and corresponding data arrays.
type SparseFunc func(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64

// SparseFuncWithN computes sparse distance that requires the total number
// of features (e.g., hamming, matching, correlation).
type SparseFuncWithN func(ind1 []int, data1 []float64, ind2 []int, data2 []float64, nFeatures int) float64

// Named returns the distance function for a given metric name.
// Returns nil if the metric is not found.
func Named(name string) Func {
	return namedDistances[name]
}

// NamedWithGrad returns the gradient distance function for a given metric name.
// Returns nil if the metric does not have a gradient version.
func NamedWithGrad(name string) GradFunc {
	return namedGradDistances[name]
}

// NamedSparse returns the sparse distance function for a given metric name.
// Returns nil if the metric does not have a sparse version.
func NamedSparse(name string) SparseFunc {
	return namedSparseDistances[name]
}

// NamedSparseWithN returns the sparse distance function requiring n_features.
// Returns nil if not applicable.
func NamedSparseWithN(name string) SparseFuncWithN {
	return namedSparseWithN[name]
}

// NamedWithParam returns the parameterized distance function for a given metric name.
// Returns nil if the metric does not have a parameterized version.
func NamedWithParam(name string) FuncWithParam {
	return namedParamDistances[name]
}

// IsValid returns true if the metric name is a known distance metric.
func IsValid(name string) bool {
	_, ok := namedDistances[name]
	if ok {
		return true
	}
	_, ok = namedParamDistances[name]
	if ok {
		return true
	}
	_, ok = namedDiscreteDistances[name]
	return ok
}

// IsDiscrete returns true if the metric operates on discrete/scalar objects
// rather than vectors.
func IsDiscrete(name string) bool {
	_, ok := namedDiscreteDistances[name]
	return ok
}

// IsSpecial returns true if the metric requires special pairwise computation
// (not compatible with standard pairwise_distances).
func IsSpecial(name string) bool {
	switch name {
	case "hellinger", "ll_dirichlet", "symmetric_kl", "poincare":
		return true
	}
	return false
}

// HasGradient returns true if the metric has a gradient version
// (needed for inverse_transform).
func HasGradient(name string) bool {
	_, ok := namedGradDistances[name]
	return ok
}

// HasSparse returns true if the metric has a sparse vector variant.
func HasSparse(name string) bool {
	_, ok := namedSparseDistances[name]
	if ok {
		return true
	}
	_, ok = namedSparseWithN[name]
	return ok
}

// SparseNeedsNFeatures returns true if the sparse variant requires n_features.
func SparseNeedsNFeatures(name string) bool {
	_, ok := namedSparseWithN[name]
	return ok
}

// namedDistances maps metric names to simple distance functions.
var namedDistances map[string]Func

// namedParamDistances maps metric names to parameterized distance functions.
var namedParamDistances map[string]FuncWithParam

// namedGradDistances maps metric names to gradient distance functions.
var namedGradDistances map[string]GradFunc

// namedSparseDistances maps metric names to sparse distance functions.
var namedSparseDistances map[string]SparseFunc

// namedSparseWithN maps metric names to sparse functions requiring n_features.
var namedSparseWithN map[string]SparseFuncWithN

// namedDiscreteDistances maps discrete metric names (kept separate because
// they operate on scalar values, not vectors).
var namedDiscreteDistances map[string]bool

func init() {
	namedDistances = map[string]Func{
		// Minkowski family
		"euclidean":   Euclidean,
		"l2":          Euclidean,
		"manhattan":   Manhattan,
		"taxicab":     Manhattan,
		"l1":          Manhattan,
		"chebyshev":   Chebyshev,
		"linfinity":   Chebyshev,
		"linfty":      Chebyshev,
		"linf":        Chebyshev,
		"canberra":    Canberra,
		"braycurtis":  BrayCurtis,
		"cosine":      Cosine,
		"correlation": Correlation,
		"hellinger":   Hellinger,
		"haversine":   Haversine,
		"poincare":    Poincare,

		// Binary metrics
		"jaccard":        Jaccard,
		"dice":           Dice,
		"hamming":        Hamming,
		"matching":       Matching,
		"kulsinski":      Kulsinski,
		"rogerstanimoto": RogersTanimoto,
		"russellrao":     RussellRao,
		"sokalsneath":    SokalSneath,
		"sokalmichener":  SokalMichener,
		"yule":           Yule,
	}

	namedParamDistances = map[string]FuncWithParam{
		"minkowski":              Minkowski,
		"seuclidean":             StandardisedEuclidean,
		"standardised_euclidean": StandardisedEuclidean,
		"wminkowski":             WeightedMinkowski,
		"weighted_minkowski":     WeightedMinkowski,
		"mahalanobis":            Mahalanobis,
		"symmetric_kl":           SymmetricKL,
		"ll_dirichlet":           LLDirichlet,
	}

	namedGradDistances = map[string]GradFunc{
		"euclidean":   EuclideanGrad,
		"l2":          EuclideanGrad,
		"manhattan":   ManhattanGrad,
		"taxicab":     ManhattanGrad,
		"l1":          ManhattanGrad,
		"chebyshev":   ChebyshevGrad,
		"linfinity":   ChebyshevGrad,
		"linfty":      ChebyshevGrad,
		"linf":        ChebyshevGrad,
		"cosine":      CosineGrad,
		"correlation": CorrelationGrad,
		"canberra":    CanberraGrad,
		"braycurtis":  BrayCurtisGrad,
		"hellinger":   HellingerGrad,
		"haversine":   HaversineGrad,
		"hyperboloid": HyperboloidGrad,
	}

	namedSparseDistances = map[string]SparseFunc{
		"euclidean":    SparseEuclidean,
		"l2":           SparseEuclidean,
		"manhattan":    SparseManhattan,
		"taxicab":      SparseManhattan,
		"l1":           SparseManhattan,
		"chebyshev":    SparseChebyshev,
		"linfinity":    SparseChebyshev,
		"linfty":       SparseChebyshev,
		"linf":         SparseChebyshev,
		"canberra":     SparseCanberra,
		"braycurtis":   SparseBrayCurtis,
		"cosine":       SparseCosine,
		"hellinger":    SparseHellinger,
		"jaccard":      SparseJaccard,
		"dice":         SparseDice,
		"sokalsneath":  SparseSokalSneath,
		"ll_dirichlet": SparseLLDirichlet,
	}

	namedSparseWithN = map[string]SparseFuncWithN{
		"hamming":        SparseHamming,
		"matching":       SparseMatching,
		"kulsinski":      SparseKulsinski,
		"rogerstanimoto": SparseRogersTanimoto,
		"russellrao":     SparseRussellRao,
		"sokalmichener":  SparseSokalMichener,
		"correlation":    SparseCorrelation,
	}

	namedDiscreteDistances = map[string]bool{
		"categorical":              true,
		"hierarchical_categorical": true,
		"ordinal":                  true,
		"count":                    true,
		"string":                   true,
	}
}
