package distance

import "math"

// Haversine computes the great circle distance between two points on a sphere.
// Input vectors must have exactly 2 elements: [latitude, longitude] in radians.
// Corresponds to distances.py haversine().
func Haversine(x, y []float64) float64 {
	sinLat := math.Sin(0.5 * (x[0] - y[0]))
	sinLon := math.Sin(0.5 * (x[1] - y[1]))
	a := sinLat*sinLat + math.Cos(x[0])*math.Cos(y[0])*sinLon*sinLon
	if a < 0 {
		a = 0
	}
	if a > 1 {
		a = 1
	}
	return 2.0 * math.Asin(math.Sqrt(a))
}

// HaversineGrad computes haversine distance and gradient w.r.t. x.
// Corresponds to distances.py haversine_grad().
func HaversineGrad(x, y []float64) (float64, []float64) {
	// The Python implementation adds pi/2 to latitude to avoid polar singularities
	// during spectral initialization.
	xLat := x[0] + math.Pi/2
	yLat := y[0] + math.Pi/2

	sinLat := math.Sin(0.5 * (xLat - yLat))
	sinLon := math.Sin(0.5 * (x[1] - y[1]))
	cosXLat := math.Cos(xLat)
	cosYLat := math.Cos(yLat)
	a := sinLat*sinLat + cosXLat*cosYLat*sinLon*sinLon
	if a < 0 {
		a = 0
	}
	if a > 1 {
		a = 1
	}
	dist := 2.0 * math.Asin(math.Sqrt(a))

	grad := make([]float64, 2)
	denom := math.Sqrt(math.Abs(a-1)) * math.Sqrt(math.Abs(a))

	sinLatHalf := math.Sin(0.5 * (x[0] - y[0]))
	cosLatHalf := math.Cos(0.5 * (x[0] - y[0]))
	sinLonHalf := math.Sin(0.5 * (x[1] - y[1]))
	cosLonHalf := math.Cos(0.5 * (x[1] - y[1]))

	grad[0] = (sinLatHalf*cosLatHalf - math.Sin(xLat)*cosYLat*sinLonHalf*sinLonHalf) / (denom + 1e-6)
	grad[1] = (cosXLat * cosYLat * sinLonHalf * cosLonHalf) / (denom + 1e-6)

	return dist, grad
}

// Poincare computes the Poincare ball distance.
// Corresponds to distances.py poincare().
func Poincare(x, y []float64) float64 {
	var sqDist, sqNormX, sqNormY float64
	for i := range x {
		d := x[i] - y[i]
		sqDist += d * d
		sqNormX += x[i] * x[i]
		sqNormY += y[i] * y[i]
	}
	denom := (1 - sqNormX) * (1 - sqNormY)
	if denom <= 0 {
		denom = 1e-10
	}
	delta := 2 * sqDist / denom
	return math.Acosh(1 + delta)
}

// HyperboloidGrad computes the hyperboloid model distance and gradient.
// This is a gradient-only metric used for embedding in hyperbolic space.
// Corresponds to distances.py hyperboloid_grad().
func HyperboloidGrad(x, y []float64) (float64, []float64) {
	var dot, sqNormX, sqNormY float64
	for i := range x {
		dot += x[i] * y[i]
		sqNormX += x[i] * x[i]
		sqNormY += y[i] * y[i]
	}
	s := math.Sqrt(1 + sqNormX)
	t := math.Sqrt(1 + sqNormY)
	b := s*t - dot
	if b <= 1 {
		b = 1 + 1e-8
	}

	dist := math.Acosh(b)
	grad := make([]float64, len(x))
	denom := math.Sqrt(b-1) * math.Sqrt(b+1)
	if denom < 1e-6 {
		denom = 1e-6
	}
	for i := range x {
		grad[i] = (x[i]*t/s - y[i]) / denom
	}
	return dist, grad
}
