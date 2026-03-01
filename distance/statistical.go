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

	distDenom := sqrtNormProd
	gradDenom := 2 * dist
	gradNumerConst := (l1Y * product) / (2 * math.Pow(distDenom, 3))

	for i := range x {
		gradTerm := math.Sqrt(x[i] * y[i])
		// UMAP-learn has y / grad_term * dist_denom which evaluates to (y / grad_term) * dist_denom
		// and it handles division by zero by numpy conventions (or fastmath).
		// We'll avoid actual div by 0 by adding 1e-6 to gradTerm if it's 0.
		if gradTerm < 1e-8 {
			gradTerm = 1e-8
		}
		grad[i] = (gradNumerConst - (y[i] / gradTerm * distDenom)) / gradDenom
	}
	return dist, grad
}

// SymmetricKL computes the symmetric KL divergence.
// Params: "z" (float64, default 1e-11) smoothing constant.
// Corresponds to distances.py symmetric_kl().
func SymmetricKL(x, y []float64, params map[string]any) float64 {
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
func LLDirichlet(x, y []float64, params map[string]any) float64 {
	n1 := 0.0
	n2 := 0.0
	for _, v := range x {
		n1 += v
	}
	for _, v := range y {
		n2 += v
	}

	logB := 0.0
	selfDenom1 := 0.0
	selfDenom2 := 0.0

	for i := range x {
		if x[i]*y[i] > 0.9 {
			logB += logBeta(x[i], y[i])
			selfDenom1 += logSingleBeta(x[i])
			selfDenom2 += logSingleBeta(y[i])
		} else {
			if x[i] > 0.9 {
				selfDenom1 += logSingleBeta(x[i])
			}
			if y[i] > 0.9 {
				selfDenom2 += logSingleBeta(y[i])
			}
		}
	}

	term1 := (1.0 / n2) * (logB - logBeta(n1, n2) - (selfDenom2 - logSingleBeta(n2)))
	term2 := (1.0 / n1) * (logB - logBeta(n2, n1) - (selfDenom1 - logSingleBeta(n1)))

	val := term1 + term2
	if val < 0 {
		val = 0
	}
	return math.Sqrt(val)
}

func logBeta(x, y float64) float64 {
	a := math.Min(x, y)
	b := math.Max(x, y)
	if b < 5 {
		value := -math.Log(b)
		for i := 1; i < int(a); i++ {
			value += math.Log(float64(i)) - math.Log(b+float64(i))
		}
		return value
	}
	return approxLogGamma(x) + approxLogGamma(y) - approxLogGamma(x+y)
}

func logSingleBeta(x float64) float64 {
	return math.Log(2.0)*(-2.0*x+0.5) + 0.5*math.Log(2.0*math.Pi/x) + 0.125/x
}

// approxLogGamma approximates log(Gamma(x)) using Stirling's approximation.
// Corresponds to distances.py approx_log_Gamma().
func approxLogGamma(x float64) float64 {
	if x == 1 {
		return 0
	}
	// Stirling's approximation as used in Python UMAP
	return x*math.Log(x) - x + 0.5*math.Log(2*math.Pi/x) + 1.0/(12.0*x)
}
