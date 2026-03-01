package distance

import "math"

// StandardisedEuclidean computes the standardised Euclidean distance.
// Params: "sigma" ([]float64) per-coordinate variance.
// Corresponds to distances.py standardised_euclidean().
func StandardisedEuclidean(x, y []float64, params map[string]any) float64 {
	sigma := getFloat64Slice(params, "sigma")
	var sum float64
	for i := range x {
		d := x[i] - y[i]
		s := 1.0
		if sigma != nil && i < len(sigma) {
			s = sigma[i]
		}
		sum += d * d / s
	}
	return math.Sqrt(sum)
}

// WeightedMinkowski computes the weighted Minkowski distance.
// Params: "w" ([]float64) per-coordinate weights, "p" (float64, default 2).
// Corresponds to distances.py weighted_minkowski().
func WeightedMinkowski(x, y []float64, params map[string]any) float64 {
	w := getFloat64Slice(params, "w")
	p := 2.0
	if v, ok := params["p"]; ok {
		p = v.(float64)
	}
	var sum float64
	for i := range x {
		wi := 1.0
		if w != nil && i < len(w) {
			wi = w[i]
		}
		sum += wi * math.Pow(math.Abs(x[i]-y[i]), p)
	}
	return math.Pow(sum, 1.0/p)
}

// Mahalanobis computes the Mahalanobis distance.
// Params: "vinv" ([][]float64) inverse covariance matrix.
// Corresponds to distances.py mahalanobis().
func Mahalanobis(x, y []float64, params map[string]any) float64 {
	vinv := getFloat64Matrix(params, "vinv")
	n := len(x)
	diff := make([]float64, n)
	for i := range n {
		diff[i] = x[i] - y[i]
	}

	var sum float64
	for i := range n {
		tmp := 0.0
		if vinv != nil && i < len(vinv) {
			for j := range n {
				if j < len(vinv[i]) {
					tmp += vinv[i][j] * diff[j]
				}
			}
		} else {
			tmp = diff[i] // identity matrix fallback
		}
		sum += tmp * diff[i]
	}

	if sum < 0 {
		sum = 0
	}
	return math.Sqrt(sum)
}

// MahalanobisGrad computes Mahalanobis distance and gradient w.r.t. x.
// Corresponds to distances.py mahalanobis_grad().
func MahalanobisGrad(x, y []float64) (float64, []float64) {
	// Without params, uses identity matrix.
	n := len(x)
	diff := make([]float64, n)
	var sum float64
	for i := range n {
		diff[i] = x[i] - y[i]
		sum += diff[i] * diff[i]
	}
	dist := math.Sqrt(sum)
	grad := make([]float64, n)
	denom := 1e-6 + dist
	for i := range n {
		grad[i] = diff[i] / denom
	}
	return dist, grad
}

func getFloat64Slice(params map[string]any, key string) []float64 {
	if params == nil {
		return nil
	}
	v, ok := params[key]
	if !ok {
		return nil
	}
	if s, ok := v.([]float64); ok {
		return s
	}
	return nil
}

func getFloat64Matrix(params map[string]any, key string) [][]float64 {
	if params == nil {
		return nil
	}
	v, ok := params[key]
	if !ok {
		return nil
	}
	if m, ok := v.([][]float64); ok {
		return m
	}
	return nil
}
