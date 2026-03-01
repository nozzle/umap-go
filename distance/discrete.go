package distance

// Discrete distance metrics for non-vector data.
// These operate on scalar values encoded in single-element float64 slices.
// Corresponds to distances.py discrete metrics.

import "math"

// CategoricalDistance returns 0 if x == y, 1 otherwise.
// Corresponds to distances.py categorical_distance().
func CategoricalDistance(x, y float64) float64 {
	if x == y {
		return 0
	}
	return 1
}

// OrdinalDistance returns |x - y| / supportSize.
// Corresponds to distances.py ordinal_distance().
func OrdinalDistance(x, y, supportSize float64) float64 {
	return math.Abs(x-y) / supportSize
}

// CountDistance computes a Poisson log-likelihood based distance.
// Corresponds to distances.py count_distance().
func CountDistance(x, y, poissonLambda, normalisation float64) float64 {
	lo := int(math.Min(x, y))
	hi := int(math.Max(x, y))
	var sum float64
	for k := lo; k < hi; k++ {
		kf := float64(k)
		sum += kf*math.Log(poissonLambda) - poissonLambda - logFactorial(kf)
	}
	return sum / normalisation
}

// LevenshteinDistance computes the edit distance between two strings
// represented as float64 slices (character codes).
// Corresponds to distances.py levenshtein().
func LevenshteinDistance(x, y []float64, normalisation float64, maxDist int) float64 {
	n := len(x)
	m := len(y)
	if math.Abs(float64(n-m)) > float64(maxDist) {
		return float64(maxDist) / normalisation
	}

	// Standard DP edit distance.
	prev := make([]int, m+1)
	curr := make([]int, m+1)
	for j := range m + 1 {
		prev[j] = j
	}
	for i := 1; i <= n; i++ {
		curr[0] = i
		for j := 1; j <= m; j++ {
			cost := 0
			if x[i-1] != y[j-1] {
				cost = 1
			}
			curr[j] = min(
				prev[j]+1,
				curr[j-1]+1,
				prev[j-1]+cost,
			)
		}
		prev, curr = curr, prev
	}
	return float64(prev[m]) / normalisation
}

func logFactorial(n float64) float64 {
	if n <= 1 {
		return 0
	}
	v, _ := math.Lgamma(n + 1)
	return v
}
