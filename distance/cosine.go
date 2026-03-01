package distance

import "math"

// Cosine computes the cosine distance: 1 - cos(x, y).
// Corresponds to distances.py cosine().
func Cosine(x, y []float64) float64 {
	var dot, normX, normY float64
	for i := range x {
		dot += x[i] * y[i]
		normX += x[i] * x[i]
		normY += y[i] * y[i]
	}
	if normX == 0 && normY == 0 {
		return 0
	}
	if normX == 0 || normY == 0 {
		return 1
	}
	return 1.0 - dot/math.Sqrt(normX*normY)
}

// CosineGrad computes cosine distance and gradient w.r.t. x.
// Corresponds to distances.py cosine_grad().
func CosineGrad(x, y []float64) (float64, []float64) {
	var dot, normX, normY float64
	for i := range x {
		dot += x[i] * y[i]
		normX += x[i] * x[i]
		normY += y[i] * y[i]
	}
	dist := 0.0
	grad := make([]float64, len(x))
	if normX == 0 && normY == 0 {
		return 0, grad
	}
	if normX == 0 || normY == 0 {
		return 1, grad
	}
	dist = 1.0 - dot/math.Sqrt(normX*normY)
	denom := math.Sqrt(normX*normX*normX) * math.Sqrt(normY)
	for i := range x {
		grad[i] = -(x[i]*dot - y[i]*normX) / denom
	}
	return dist, grad
}

// Correlation computes the correlation distance: 1 - Pearson correlation.
// Corresponds to distances.py correlation().
func Correlation(x, y []float64) float64 {
	n := float64(len(x))
	if n == 0 {
		return 0
	}

	var muX, muY float64
	for i := range x {
		muX += x[i]
		muY += y[i]
	}
	muX /= n
	muY /= n

	var dot, normX, normY float64
	for i := range x {
		dx := x[i] - muX
		dy := y[i] - muY
		dot += dx * dy
		normX += dx * dx
		normY += dy * dy
	}

	if normX == 0 && normY == 0 {
		return 0
	}
	if dot == 0 {
		return 1
	}
	return 1.0 - dot/math.Sqrt(normX*normY)
}

// CorrelationGrad computes correlation distance and gradient w.r.t. x.
// Corresponds to distances.py correlation_grad().
func CorrelationGrad(x, y []float64) (float64, []float64) {
	n := float64(len(x))
	grad := make([]float64, len(x))
	if n == 0 {
		return 0, grad
	}

	var muX, muY float64
	for i := range x {
		muX += x[i]
		muY += y[i]
	}
	muX /= n
	muY /= n

	var dot, normX, normY float64
	for i := range x {
		dx := x[i] - muX
		dy := y[i] - muY
		dot += dx * dy
		normX += dx * dx
		normY += dy * dy
	}

	if normX == 0 && normY == 0 {
		return 0, grad
	}
	if dot == 0 {
		return 1, grad
	}

	dist := 1.0 - dot/math.Sqrt(normX*normY)
	sqNormX := math.Sqrt(normX)
	for i := range x {
		dx := x[i] - muX
		dy := y[i] - muY
		grad[i] = ((dx/sqNormX - dy/dot) * dist)
		_ = dy // used above
	}
	return dist, grad
}
