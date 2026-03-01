package umap

// curves.go implements find_ab_params: curve fitting to find the a, b parameters
// for the UMAP membership function.
//
// Corresponds to umap_.py find_ab_params().
//
// The function fits: y = 1 / (1 + a * x^(2b))
// to the target curve derived from spread and min_dist parameters.

import (
	"math"
)

// FindABParams computes the a, b parameters for the UMAP membership function
// via curve fitting (Levenberg-Marquardt style).
//
// The target curve is:
//
//	y = 1.0  if x <= min_dist
//	y = exp(-(x - min_dist) / (spread - min_dist))  otherwise
//
// And we fit: y = 1 / (1 + a * x^(2b))
//
// Returns (a, b).
func FindABParams(spread, minDist float64) (float64, float64) {
	// Generate target curve — matches Python's np.linspace(0, 3*spread, 300)
	const nPoints = 300
	xv := make([]float64, nPoints)
	yv := make([]float64, nPoints)

	for i := range nPoints {
		x := 3.0 * spread * float64(i) / float64(nPoints-1)
		xv[i] = x
		if x < minDist {
			yv[i] = 1.0
		} else {
			yv[i] = math.Exp(-(x - minDist) / spread)
		}
	}

	// Levenberg-Marquardt curve fitting
	// Model: f(x; a, b) = 1 / (1 + a * x^(2b))
	// We minimize sum of squared residuals.
	//
	// scipy.optimize.curve_fit defaults: p0=[1,1], maxfev=800 (for 2 params)
	a, b := 1.0, 1.0

	const maxIter = 200
	lambda := 1e-3 // damping parameter
	const lambdaUp = 10.0
	const lambdaDown = 10.0

	prevCost := lmCost(xv, yv, a, b)

	for range maxIter {
		// Compute Jacobian^T * residuals and Jacobian^T * Jacobian (2x2 system)
		var j11, j12, j22 float64 // J^T J entries
		var g1, g2 float64        // J^T r (gradient)

		for i := range nPoints {
			x := xv[i]
			if x == 0 {
				continue // derivative is 0 at x=0
			}
			x2b := math.Pow(x, 2*b)
			denom := 1.0 + a*x2b
			pred := 1.0 / denom
			r := pred - yv[i] // residual

			// Partial derivatives of f w.r.t. a, b:
			// df/da = -x^(2b) / (1 + a*x^(2b))^2
			// df/db = -2*a*x^(2b)*ln(x) / (1 + a*x^(2b))^2
			denom2 := denom * denom
			dfda := -x2b / denom2
			dfdb := -2.0 * a * x2b * math.Log(x) / denom2

			j11 += dfda * dfda
			j12 += dfda * dfdb
			j22 += dfdb * dfdb
			g1 += dfda * r
			g2 += dfdb * r
		}

		// Solve (J^T J + lambda * diag(J^T J)) * delta = -J^T r
		// 2x2 system: [[j11+lambda*j11, j12], [j12, j22+lambda*j22]] * [da, db] = [-g1, -g2]
		a11 := j11 * (1 + lambda)
		a12 := j12
		a22 := j22 * (1 + lambda)

		det := a11*a22 - a12*a12
		if math.Abs(det) < 1e-30 {
			break
		}

		da := (-g1*a22 - (-g2)*a12) / det
		db := (a11*(-g2) - a12*(-g1)) / det

		newA := a + da
		newB := b + db

		newCost := lmCost(xv, yv, newA, newB)

		if newCost < prevCost {
			// Accept step
			a = newA
			b = newB
			prevCost = newCost
			lambda /= lambdaDown

			// Check convergence
			if math.Abs(da) < 1e-12 && math.Abs(db) < 1e-12 {
				break
			}
		} else {
			// Reject step, increase damping
			lambda *= lambdaUp
		}
	}

	return a, b
}

// lmCost computes the sum of squared residuals.
func lmCost(xv, yv []float64, a, b float64) float64 {
	var sse float64
	for i := range xv {
		pred := 1.0 / (1.0 + a*math.Pow(xv[i], 2*b))
		diff := pred - yv[i]
		sse += diff * diff
	}
	return sse
}
