// Package rand provides a random number source abstraction for UMAP.
//
// In production, Source wraps math/rand/v2. In tests, a Recorded source
// plays back prerecorded sequences from the Python reference implementation,
// enabling byte-for-byte reproducibility verification.
package rand

// Source provides all random operations UMAP needs.
type Source interface {
	// Intn returns a random int in [0, n).
	Intn(n int) int

	// Float64 returns a random float64 in [0.0, 1.0).
	Float64() float64

	// NormFloat64 returns a normally distributed float64 (mean=0, stddev=1).
	NormFloat64() float64

	// Perm returns a random permutation of [0, n).
	Perm(n int) []int

	// UniformFloat64 returns a random float64 in [low, high).
	UniformFloat64(low, high float64) float64
}
