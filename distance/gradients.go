package distance

// Gradient versions for special output metric embeddings.
// These are used in inverse_transform for non-standard embedding spaces.
// Corresponds to distances.py gradient-only metrics.

import "math"

// SphericalGaussianEnergyGrad computes the spherical Gaussian energy distance
// and gradient. Embedding is [x, y, sigma].
// Corresponds to distances.py spherical_gaussian_energy_grad().
func SphericalGaussianEnergyGrad(x, y []float64) (float64, []float64) {
	mu1 := x[0] - y[0]
	mu2 := x[1] - y[1]
	sigma := math.Abs(x[2]) + math.Abs(y[2])

	if sigma == 0 {
		return 10.0, []float64{0, 0, -1}
	}

	dist := (mu1*mu1+mu2*mu2)/(2*sigma) + math.Log(sigma) + math.Log(2*math.Pi)

	grad := make([]float64, 3)
	grad[0] = mu1 / sigma
	grad[1] = mu2 / sigma
	signX2 := 1.0
	if x[2] < 0 {
		signX2 = -1.0
	}
	grad[2] = signX2 * (-(mu1*mu1+mu2*mu2)/(2*sigma*sigma) + 1.0/sigma)

	return dist, grad
}

// DiagonalGaussianEnergyGrad computes the diagonal Gaussian energy distance
// and gradient. Embedding is [x, y, sigma_x, sigma_y].
// Corresponds to distances.py diagonal_gaussian_energy_grad().
func DiagonalGaussianEnergyGrad(x, y []float64) (float64, []float64) {
	mu1 := x[0] - y[0]
	mu2 := x[1] - y[1]
	sigma11 := math.Abs(x[2]) + math.Abs(y[2])
	sigma22 := math.Abs(x[3]) + math.Abs(y[3])

	det := sigma11 * sigma22
	if det == 0 {
		return 10.0, []float64{0, 0, -1, -1}
	}

	mDist := math.Abs(sigma22)*mu1*mu1 + math.Abs(sigma11)*mu2*mu2
	dist := (mDist/det+math.Log(math.Abs(det)))/2.0 + math.Log(2*math.Pi)

	grad := make([]float64, 4)
	grad[0] = math.Abs(sigma22) * mu1 / det
	grad[1] = math.Abs(sigma11) * mu2 / det

	signX2 := 1.0
	if x[2] < 0 {
		signX2 = -1.0
	}
	signX3 := 1.0
	if x[3] < 0 {
		signX3 = -1.0
	}
	grad[2] = signX2 * (mu2*mu2/det - mDist*sigma22/(det*det) + sigma22/det) / 2.0
	grad[3] = signX3 * (mu1*mu1/det - mDist*sigma11/(det*det) + sigma11/det) / 2.0

	return dist, grad
}

// GaussianEnergyGrad computes the full 2D Gaussian energy distance
// and gradient. Embedding is [x, y, width, height, angle].
// Corresponds to distances.py gaussian_energy_grad().
func GaussianEnergyGrad(x, y []float64) (float64, []float64) {
	mu1 := x[0] - y[0]
	mu2 := x[1] - y[1]

	// Build rotation-based covariance from width, height, angle
	w1 := math.Abs(x[2])
	h1 := math.Abs(x[3])
	a1 := math.Asin(math.Sin(x[4])) // normalize angle

	w2 := math.Abs(y[2])
	h2 := math.Abs(y[3])
	a2 := math.Asin(math.Sin(y[4]))

	// Covariance matrices
	cos1, sin1 := math.Cos(a1), math.Sin(a1)
	cos2, sin2 := math.Cos(a2), math.Sin(a2)

	s11 := w1*cos1*cos1 + h1*sin1*sin1 + w2*cos2*cos2 + h2*sin2*sin2
	s12 := (w1-h1)*cos1*sin1 + (w2-h2)*cos2*sin2
	s22 := w1*sin1*sin1 + h1*cos1*cos1 + w2*sin2*sin2 + h2*cos2*cos2

	det := s11*s22 - s12*s12
	if math.Abs(det) < 1e-32 {
		return 10.0, []float64{0, 0, -1, -1, 0}
	}

	// Inverse of sum covariance
	invDet := 1.0 / det
	xInvSigmaY := invDet * (s22*mu1*mu1 - 2*s12*mu1*mu2 + s11*mu2*mu2)

	dist := (xInvSigmaY+math.Log(math.Abs(det)))/2.0 + math.Log(2*math.Pi)

	grad := make([]float64, 5)
	grad[0] = invDet * (s22*mu1 - s12*mu2)
	grad[1] = invDet * (s11*mu2 - s12*mu1)

	// Gradients w.r.t. width, height, angle are complex
	// Simplified version - compute partial derivatives of det and xInvSigmaY
	signW := 1.0
	if x[2] < 0 {
		signW = -1.0
	}
	signH := 1.0
	if x[3] < 0 {
		signH = -1.0
	}

	dS11dW := signW * cos1 * cos1
	dS11dH := signH * sin1 * sin1
	dS22dW := signW * sin1 * sin1
	dS22dH := signH * cos1 * cos1
	dS12dW := signW * cos1 * sin1
	dS12dH := -signH * cos1 * sin1

	for _, entry := range [2]struct {
		idx  int
		ds11 float64
		ds22 float64
		ds12 float64
	}{
		{2, dS11dW, dS22dW, dS12dW},
		{3, dS11dH, dS22dH, dS12dH},
	} {
		dDet := entry.ds11*s22 + s11*entry.ds22 - 2*s12*entry.ds12
		dXISY := invDet * (entry.ds22*mu1*mu1 - 2*entry.ds12*mu1*mu2 + entry.ds11*mu2*mu2 - xInvSigmaY*dDet)
		grad[entry.idx] = (dXISY + dDet/det) / 2.0
	}

	// Gradient w.r.t. angle
	cosA := math.Cos(x[4])
	dA := cosA // derivative of asin(sin(x)) = cos(x) when in range
	dS11dA := (-w1*2*cos1*sin1 + h1*2*sin1*cos1) * dA
	dS22dA := (w1*2*sin1*cos1 - h1*2*cos1*sin1) * dA
	dS12dA := ((w1 - h1) * (cos1*cos1 - sin1*sin1)) * dA
	dDet := dS11dA*s22 + s11*dS22dA - 2*s12*dS12dA
	dXISY := invDet * (dS22dA*mu1*mu1 - 2*dS12dA*mu1*mu2 + dS11dA*mu2*mu2 - xInvSigmaY*dDet)
	grad[4] = (dXISY + dDet/det) / 2.0

	return dist, grad
}

// SphericalGaussianGrad computes the spherical Gaussian distance and gradient.
// Distinct from the energy version. Embedding is [x, y, sigma].
// Corresponds to distances.py spherical_gaussian_grad().
func SphericalGaussianGrad(x, y []float64) (float64, []float64) {
	mu1 := x[0] - y[0]
	mu2 := x[1] - y[1]
	sigma := x[2] + y[2]

	if sigma == 0 {
		return 10.0, []float64{0, 0, -1}
	}

	absSigma := math.Abs(sigma)
	dist := (mu1*mu1+mu2*mu2)/absSigma + 2*math.Log(absSigma) + math.Log(2*math.Pi)

	grad := make([]float64, 3)
	grad[0] = 2 * mu1 / absSigma
	grad[1] = 2 * mu2 / absSigma
	grad[2] = -(mu1*mu1+mu2*mu2)/(absSigma*absSigma) + 2.0/absSigma
	if sigma < 0 {
		grad[2] = -grad[2]
	}

	return dist, grad
}
