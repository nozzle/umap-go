package distance

import "math"

// Euclidean computes the Euclidean (L2) distance.
// Corresponds to distances.py euclidean().
func Euclidean(x, y []float64) float64 {
	var sum float64
	for i := range x {
		d := x[i] - y[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// EuclideanGrad computes Euclidean distance and gradient w.r.t. x.
// Corresponds to distances.py euclidean_grad().
func EuclideanGrad(x, y []float64) (float64, []float64) {
	dist := Euclidean(x, y)
	grad := make([]float64, len(x))
	denom := 1e-6 + dist
	for i := range x {
		grad[i] = (x[i] - y[i]) / denom
	}
	return dist, grad
}

// Manhattan computes the Manhattan (L1/taxicab) distance.
// Corresponds to distances.py manhattan().
func Manhattan(x, y []float64) float64 {
	var sum float64
	for i := range x {
		sum += math.Abs(x[i] - y[i])
	}
	return sum
}

// ManhattanGrad computes Manhattan distance and gradient w.r.t. x.
// Corresponds to distances.py manhattan_grad().
func ManhattanGrad(x, y []float64) (float64, []float64) {
	dist := Manhattan(x, y)
	grad := make([]float64, len(x))
	for i := range x {
		d := x[i] - y[i]
		if d > 0 {
			grad[i] = 1.0
		} else if d < 0 {
			grad[i] = -1.0
		}
	}
	return dist, grad
}

// Chebyshev computes the Chebyshev (L∞) distance.
// Corresponds to distances.py chebyshev().
func Chebyshev(x, y []float64) float64 {
	var maxDist float64
	for i := range x {
		d := math.Abs(x[i] - y[i])
		if d > maxDist {
			maxDist = d
		}
	}
	return maxDist
}

// ChebyshevGrad computes Chebyshev distance and gradient w.r.t. x.
// Corresponds to distances.py chebyshev_grad().
func ChebyshevGrad(x, y []float64) (float64, []float64) {
	dist := 0.0
	maxIdx := 0
	for i := range x {
		d := math.Abs(x[i] - y[i])
		if d > dist {
			dist = d
			maxIdx = i
		}
	}
	grad := make([]float64, len(x))
	d := x[maxIdx] - y[maxIdx]
	if d > 0 {
		grad[maxIdx] = 1.0
	} else if d < 0 {
		grad[maxIdx] = -1.0
	}
	return dist, grad
}

// Minkowski computes the Minkowski distance with parameter p.
// Params: "p" (float64, default 2).
// Corresponds to distances.py minkowski().
func Minkowski(x, y []float64, params map[string]interface{}) float64 {
	p := 2.0
	if v, ok := params["p"]; ok {
		p = v.(float64)
	}
	var sum float64
	for i := range x {
		sum += math.Pow(math.Abs(x[i]-y[i]), p)
	}
	return math.Pow(sum, 1.0/p)
}

// Canberra computes the Canberra distance.
// Corresponds to distances.py canberra().
func Canberra(x, y []float64) float64 {
	var sum float64
	for i := range x {
		denom := math.Abs(x[i]) + math.Abs(y[i])
		if denom > 0 {
			sum += math.Abs(x[i]-y[i]) / denom
		}
	}
	return sum
}

// CanberraGrad computes Canberra distance and gradient w.r.t. x.
// Corresponds to distances.py canberra_grad().
func CanberraGrad(x, y []float64) (float64, []float64) {
	dist := 0.0
	grad := make([]float64, len(x))
	for i := range x {
		denom := math.Abs(x[i]) + math.Abs(y[i])
		if denom > 0 {
			diff := math.Abs(x[i] - y[i])
			dist += diff / denom
			sign := 1.0
			if x[i]-y[i] < 0 {
				sign = -1.0
			}
			signX := 1.0
			if x[i] < 0 {
				signX = -1.0
			}
			grad[i] = sign/denom - diff*signX/(denom*denom)
		}
	}
	return dist, grad
}

// BrayCurtis computes the Bray-Curtis distance.
// Corresponds to distances.py bray_curtis().
func BrayCurtis(x, y []float64) float64 {
	var numerator, denominator float64
	for i := range x {
		numerator += math.Abs(x[i] - y[i])
		denominator += math.Abs(x[i] + y[i])
	}
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

// BrayCurtisGrad computes Bray-Curtis distance and gradient w.r.t. x.
// Corresponds to distances.py bray_curtis_grad().
func BrayCurtisGrad(x, y []float64) (float64, []float64) {
	var numerator, denominator float64
	for i := range x {
		numerator += math.Abs(x[i] - y[i])
		denominator += math.Abs(x[i] + y[i])
	}
	dist := 0.0
	grad := make([]float64, len(x))
	if denominator > 0 {
		dist = numerator / denominator
		for i := range x {
			sign := 1.0
			if x[i]-y[i] < 0 {
				sign = -1.0
			}
			grad[i] = (sign - dist) / denominator
		}
	}
	return dist, grad
}
