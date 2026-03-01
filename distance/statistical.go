package distance

import "math"

// Hellinger computes the Hellinger distance between two non-negative vectors.
// Corresponds to distances.py hellinger().
func Hellinger(x, y []float64) float64 {
	var l1X, l1Y, product float64
	for i := range x {
		l1X += x[i]
		l1Y += y[i]
	}
	if l1X == 0 && l1Y == 0 {
		return 0
	}
	if l1X == 0 || l1Y == 0 {
		return 1
	}
	for i := range x {
		product += math.Sqrt(x[i] * y[i])
	}
	sqrtNormProd := math.Sqrt(l1X * l1Y)
	val := 1.0 - product/sqrtNormProd
	if val < 0 {
		val = 0
	}
	return math.Sqrt(val)
}

// HellingerGrad computes Hellinger distance and gradient w.r.t. x.
// Corresponds to distances.py hellinger_grad().
func HellingerGrad(x, y []float64) (float64, []float64) {
	var l1X, l1Y, product float64
	for i := range x {
		l1X += x[i]
		l1Y += y[i]
	}
	grad := make([]float64, len(x))
	if l1X == 0 && l1Y == 0 {
		return 0, grad
	}
	if l1X == 0 || l1Y == 0 {
		return 1, grad
	}
	for i := range x {
		product += math.Sqrt(x[i] * y[i])
	}
	sqrtNormProd := math.Sqrt(l1X * l1Y)
	val := 1.0 - product/sqrtNormProd
	if val < 0 {
		val = 0
	}
	dist := math.Sqrt(val)
	if dist < 1e-6 {
		return dist, grad
	}

	for i := range x {
		sqrtXY := math.Sqrt(x[i]*y[i]) + 1e-6
		grad[i] = (0.5*y[i]/sqrtXY/sqrtNormProd - 0.5*product/(l1X*sqrtNormProd)) / (2.0 * dist)
	}
	return dist, grad
}

// SymmetricKL computes the symmetric KL divergence.
// Params: "z" (float64, default 1e-11) smoothing constant.
// Corresponds to distances.py symmetric_kl().
func SymmetricKL(x, y []float64, params map[string]interface{}) float64 {
	z := 1e-11
	if v, ok := params["z"]; ok {
		z = v.(float64)
	}

	n := len(x)
	// Normalize to probability distributions with smoothing.
	// Note: Python version mutates inputs. We copy to avoid that.
	px := make([]float64, n)
	py := make([]float64, n)
	var sumX, sumY float64
	for i := range n {
		px[i] = x[i] + z
		py[i] = y[i] + z
		sumX += px[i]
		sumY += py[i]
	}
	for i := range n {
		px[i] /= sumX
		py[i] /= sumY
	}

	var kl1, kl2 float64
	for i := range n {
		kl1 += px[i] * math.Log(px[i]/py[i])
		kl2 += py[i] * math.Log(py[i]/px[i])
	}
	return (kl1 + kl2) / 2.0
}

// LLDirichlet computes the log-likelihood Dirichlet distance.
// Corresponds to distances.py ll_dirichlet().
func LLDirichlet(x, y []float64, params map[string]interface{}) float64 {
	n1 := float64(len(x))
	n2 := float64(len(y))

	logBetaXY := 0.0
	selfDenom1 := 0.0
	selfDenom2 := 0.0
	for i := range x {
		logBetaXY += approxLogGamma(x[i]) + approxLogGamma(y[i]) - approxLogGamma(x[i]+y[i])
		selfDenom1 += 2*approxLogGamma(x[i]) - approxLogGamma(2*x[i])
		selfDenom2 += 2*approxLogGamma(y[i]) - approxLogGamma(2*y[i])
	}

	logBetaN1N2 := n1*(approxLogGamma(n2/n1)+approxLogGamma(n2/n1)) - n1*approxLogGamma(2*n2/n1)
	logBetaN2N1 := n2*(approxLogGamma(n1/n2)+approxLogGamma(n1/n2)) - n2*approxLogGamma(2*n1/n2)
	logSingleBetaN1 := n1 * (2*approxLogGamma(1) - approxLogGamma(2))
	logSingleBetaN2 := n2 * (2*approxLogGamma(1) - approxLogGamma(2))

	term1 := (1 / n2) * (logBetaXY - logBetaN1N2 - (selfDenom2 - logSingleBetaN2))
	term2 := (1 / n1) * (logBetaXY - logBetaN2N1 - (selfDenom1 - logSingleBetaN1))

	val := term1 + term2
	if val < 0 {
		val = 0
	}
	return math.Sqrt(val)
}

// approxLogGamma approximates log(Gamma(x)) using Stirling's approximation
// for large x and math.Lgamma for small x.
// Corresponds to distances.py approx_log_Gamma().
func approxLogGamma(x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x < 20 {
		v, _ := math.Lgamma(x)
		return v
	}
	// Stirling's approximation
	return x*math.Log(x) - x + 0.5*math.Log(2*math.Pi/x) +
		1.0/(12*x) - 1.0/(360*x*x*x)
}
