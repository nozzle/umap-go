package distance

// Sparse distance metrics. Each operates on two sparse vectors represented as
// sorted index arrays and corresponding data arrays (i.e., one row from a CSR
// matrix). Corresponds to umap/sparse.py (v0.5.11).

import "math"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// sparseMerge walks two sorted index slices and calls fn for each aligned pair.
// One of the values may be zero when only one vector has a nonzero at that index.
func sparseMerge(ind1 []int, data1 []float64, ind2 []int, data2 []float64,
	fn func(v1, v2 float64)) {
	i, j := 0, 0
	for i < len(ind1) && j < len(ind2) {
		if ind1[i] == ind2[j] {
			fn(data1[i], data2[j])
			i++
			j++
		} else if ind1[i] < ind2[j] {
			fn(data1[i], 0)
			i++
		} else {
			fn(0, data2[j])
			j++
		}
	}
	for ; i < len(ind1); i++ {
		fn(data1[i], 0)
	}
	for ; j < len(ind2); j++ {
		fn(0, data2[j])
	}
}

// ---------------------------------------------------------------------------
// SparseFunc metrics (no n_features required)
// ---------------------------------------------------------------------------

// SparseEuclidean computes the Euclidean distance between two sparse vectors.
// Corresponds to sparse.py sparse_euclidean().
func SparseEuclidean(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var sum float64
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		d := v1 - v2
		sum += d * d
	})
	return math.Sqrt(sum)
}

// SparseManhattan computes the Manhattan distance between two sparse vectors.
// Corresponds to sparse.py sparse_manhattan().
func SparseManhattan(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var sum float64
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		sum += math.Abs(v1 - v2)
	})
	return sum
}

// SparseChebyshev computes the Chebyshev distance between two sparse vectors.
// Corresponds to sparse.py sparse_chebyshev().
func SparseChebyshev(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var maxDist float64
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		d := math.Abs(v1 - v2)
		if d > maxDist {
			maxDist = d
		}
	})
	return maxDist
}

// SparseCanberra computes the Canberra distance between two sparse vectors.
// Corresponds to sparse.py sparse_canberra().
func SparseCanberra(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var sum float64
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		denom := math.Abs(v1) + math.Abs(v2)
		if denom > 0 {
			sum += math.Abs(v1-v2) / denom
		}
	})
	return sum
}

// SparseBrayCurtis computes the Bray-Curtis distance between two sparse vectors.
// Corresponds to sparse.py sparse_bray_curtis().
func SparseBrayCurtis(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var numerator, denominator float64
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		numerator += math.Abs(v1 - v2)
		denominator += math.Abs(v1 + v2)
	})
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

// SparseCosine computes the cosine distance between two sparse vectors.
// Corresponds to sparse.py sparse_cosine().
func SparseCosine(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var dot, normX, normY float64
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		dot += v1 * v2
		normX += v1 * v1
		normY += v2 * v2
	})
	if normX == 0 && normY == 0 {
		return 0
	}
	if normX == 0 || normY == 0 {
		return 1
	}
	return 1.0 - dot/math.Sqrt(normX*normY)
}

// SparseHellinger computes the Hellinger distance between two sparse vectors.
// Both vectors must be non-negative.
// Corresponds to sparse.py sparse_hellinger().
func SparseHellinger(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var l1X, l1Y, product float64

	// L1 norms: sum of all nonzero entries.
	for _, v := range data1 {
		l1X += v
	}
	for _, v := range data2 {
		l1Y += v
	}
	if l1X == 0 && l1Y == 0 {
		return 0
	}
	if l1X == 0 || l1Y == 0 {
		return 1
	}

	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		if v1 > 0 && v2 > 0 {
			product += math.Sqrt(v1 * v2)
		}
	})

	sqrtNormProd := math.Sqrt(l1X * l1Y)
	val := 1.0 - product/sqrtNormProd
	if val < 0 {
		val = 0
	}
	return math.Sqrt(val)
}

// SparseJaccard computes the Jaccard distance between two sparse binary vectors.
// Corresponds to sparse.py sparse_jaccard().
func SparseJaccard(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var ntt, nne int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		xb := v1 != 0
		yb := v2 != 0
		if xb && yb {
			ntt++
		}
		if xb != yb {
			nne++
		}
	})
	if nne == 0 {
		return 0
	}
	return float64(nne) / float64(ntt+nne)
}

// SparseDice computes the Dice distance between two sparse binary vectors.
// Corresponds to sparse.py sparse_dice().
func SparseDice(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var ntt, nne int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		xb := v1 != 0
		yb := v2 != 0
		if xb && yb {
			ntt++
		}
		if xb != yb {
			nne++
		}
	})
	if nne == 0 {
		return 0
	}
	return float64(nne) / float64(2*ntt+nne)
}

// SparseSokalSneath computes the Sokal-Sneath distance between two sparse binary vectors.
// Corresponds to sparse.py sparse_sokal_sneath().
func SparseSokalSneath(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	var ntt, nne int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		xb := v1 != 0
		yb := v2 != 0
		if xb && yb {
			ntt++
		}
		if xb != yb {
			nne++
		}
	})
	if nne == 0 {
		return 0
	}
	return float64(nne) / (0.5*float64(ntt) + float64(nne))
}

// SparseLLDirichlet computes the log-likelihood Dirichlet distance for sparse vectors.
// Corresponds to sparse.py sparse_ll_dirichlet().
func SparseLLDirichlet(ind1 []int, data1 []float64, ind2 []int, data2 []float64) float64 {
	n1 := 0.0
	for _, v := range data1 {
		n1 += v
	}
	n2 := 0.0
	for _, v := range data2 {
		n2 += v
	}
	if n1 == 0 && n2 == 0 {
		return 0.0
	} else if n1 == 0 || n2 == 0 {
		return 1e8
	}

	logB := 0.0
	i1, i2 := 0, 0
	for i1 < len(ind1) && i2 < len(ind2) {
		j1, j2 := ind1[i1], ind2[i2]
		if j1 == j2 {
			if data1[i1]*data2[i2] != 0 {
				logB += logBeta(data1[i1], data2[i2])
			}
			i1++
			i2++
		} else if j1 < j2 {
			i1++
		} else {
			i2++
		}
	}

	selfDenom1 := 0.0
	for _, d1 := range data1 {
		selfDenom1 += logSingleBeta(d1)
	}

	selfDenom2 := 0.0
	for _, d2 := range data2 {
		selfDenom2 += logSingleBeta(d2)
	}

	term1 := (1.0 / n2) * (logB - logBeta(n1, n2) - (selfDenom2 - logSingleBeta(n2)))
	term2 := (1.0 / n1) * (logB - logBeta(n2, n1) - (selfDenom1 - logSingleBeta(n1)))

	val := term1 + term2
	// Python does not cap to 0. It lets math.Sqrt return NaN for negative values.
	return math.Sqrt(val)
}

// ---------------------------------------------------------------------------
// SparseFuncWithN metrics (require n_features)
// ---------------------------------------------------------------------------

// SparseHamming computes the Hamming distance between two sparse vectors.
// nFeatures is the total dimensionality.
// Corresponds to sparse.py sparse_hamming().
func SparseHamming(ind1 []int, data1 []float64, ind2 []int, data2 []float64, nFeatures int) float64 {
	if nFeatures == 0 {
		return 0
	}
	var nne int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		if v1 != v2 {
			nne++
		}
	})
	return float64(nne) / float64(nFeatures)
}

// SparseMatching computes the matching distance between two sparse binary vectors.
// nFeatures is the total dimensionality.
// Corresponds to sparse.py sparse_matching().
func SparseMatching(ind1 []int, data1 []float64, ind2 []int, data2 []float64, nFeatures int) float64 {
	if nFeatures == 0 {
		return 0
	}
	var nne int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		xb := v1 != 0
		yb := v2 != 0
		if xb != yb {
			nne++
		}
	})
	return float64(nne) / float64(nFeatures)
}

// SparseKulsinski computes the Kulsinski distance between two sparse binary vectors.
// nFeatures is the total dimensionality.
// Corresponds to sparse.py sparse_kulsinski().
func SparseKulsinski(ind1 []int, data1 []float64, ind2 []int, data2 []float64, nFeatures int) float64 {
	n := nFeatures
	var ntt, nne int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		xb := v1 != 0
		yb := v2 != 0
		if xb && yb {
			ntt++
		}
		if xb != yb {
			nne++
		}
	})
	if nne == 0 {
		return 0
	}
	return float64(nne-ntt+n) / float64(nne+n)
}

// SparseRogersTanimoto computes the Rogers-Tanimoto distance between two sparse binary vectors.
// nFeatures is the total dimensionality.
// Corresponds to sparse.py sparse_rogers_tanimoto().
func SparseRogersTanimoto(ind1 []int, data1 []float64, ind2 []int, data2 []float64, nFeatures int) float64 {
	n := nFeatures
	var nne int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		xb := v1 != 0
		yb := v2 != 0
		if xb != yb {
			nne++
		}
	})
	return float64(2*nne) / float64(n+nne)
}

// SparseRussellRao computes the Russell-Rao distance between two sparse binary vectors.
// nFeatures is the total dimensionality.
// Corresponds to sparse.py sparse_russellrao().
func SparseRussellRao(ind1 []int, data1 []float64, ind2 []int, data2 []float64, nFeatures int) float64 {
	n := nFeatures
	if n == 0 {
		return 0
	}
	var ntt, nx, ny int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		xb := v1 != 0
		yb := v2 != 0
		if xb && yb {
			ntt++
		}
		if xb {
			nx++
		}
		if yb {
			ny++
		}
	})
	if ntt == nx && ntt == ny {
		return 0
	}
	return float64(n-ntt) / float64(n)
}

// SparseSokalMichener computes the Sokal-Michener distance between two sparse binary vectors.
// nFeatures is the total dimensionality.
// Corresponds to sparse.py sparse_sokal_michener().
func SparseSokalMichener(ind1 []int, data1 []float64, ind2 []int, data2 []float64, nFeatures int) float64 {
	n := nFeatures
	var nne int
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		xb := v1 != 0
		yb := v2 != 0
		if xb != yb {
			nne++
		}
	})
	return float64(2*nne) / float64(n+nne)
}

// SparseCorrelation computes the correlation distance between two sparse vectors.
// nFeatures is the total dimensionality (needed to compute means over full dense vectors).
// Corresponds to sparse.py sparse_correlation().
func SparseCorrelation(ind1 []int, data1 []float64, ind2 []int, data2 []float64, nFeatures int) float64 {
	n := float64(nFeatures)
	if n == 0 {
		return 0
	}

	// Means computed over the full dense vector (zeros for missing entries).
	var sumX, sumY float64
	for _, v := range data1 {
		sumX += v
	}
	for _, v := range data2 {
		sumY += v
	}
	muX := sumX / n
	muY := sumY / n

	// dot(x-muX, y-muY) = dot(x,y) - muY*sum(x) - muX*sum(y) + n*muX*muY
	// norm(x-muX)^2 = sum(x^2) - 2*muX*sum(x) + n*muX^2
	var dotXY, sumSqX, sumSqY float64
	sparseMerge(ind1, data1, ind2, data2, func(v1, v2 float64) {
		dotXY += v1 * v2
		sumSqX += v1 * v1
		sumSqY += v2 * v2
	})

	covXY := dotXY - muY*sumX - muX*sumY + n*muX*muY
	normX := sumSqX - 2*muX*sumX + n*muX*muX
	normY := sumSqY - 2*muY*sumY + n*muY*muY

	if normX == 0 && normY == 0 {
		return 0
	}
	if covXY == 0 {
		return 1
	}
	return 1.0 - covXY/math.Sqrt(normX*normY)
}
