package distance

// Binary distance metrics. All treat nonzero values as true/present.
// Corresponds to distances.py binary metrics.

// Jaccard computes the Jaccard distance.
func Jaccard(x, y []float64) float64 {
	var ntt, nne int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb && yb {
			ntt++
		}
		if xb != yb {
			nne++
		}
	}
	if nne == 0 {
		return 0
	}
	return float64(nne) / float64(ntt+nne)
}

// Dice computes the Dice distance.
func Dice(x, y []float64) float64 {
	var ntt, nne int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb && yb {
			ntt++
		}
		if xb != yb {
			nne++
		}
	}
	if nne == 0 {
		return 0
	}
	return float64(nne) / float64(2*ntt+nne)
}

// Hamming computes the Hamming distance (fraction of differing elements).
func Hamming(x, y []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	var nne int
	for i := range x {
		if x[i] != y[i] {
			nne++
		}
	}
	return float64(nne) / float64(n)
}

// Matching computes the matching distance (same as Hamming on boolean data).
func Matching(x, y []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	var nne int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb != yb {
			nne++
		}
	}
	return float64(nne) / float64(n)
}

// Kulsinski computes the Kulsinski distance.
func Kulsinski(x, y []float64) float64 {
	n := len(x)
	var ntt, nne int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb && yb {
			ntt++
		}
		if xb != yb {
			nne++
		}
	}
	if nne == 0 {
		return 0
	}
	return float64(nne-ntt+n) / float64(nne+n)
}

// RogersTanimoto computes the Rogers-Tanimoto distance.
func RogersTanimoto(x, y []float64) float64 {
	n := len(x)
	var nne int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb != yb {
			nne++
		}
	}
	return float64(2*nne) / float64(n+nne)
}

// RussellRao computes the Russell-Rao distance.
func RussellRao(x, y []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	var ntt, nx, ny int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb && yb {
			ntt++
		}
		if xb {
			nx++
		}
		if yb {
			ny++
		}
	}
	if ntt == nx && ntt == ny {
		return 0
	}
	return float64(n-ntt) / float64(n)
}

// SokalSneath computes the Sokal-Sneath distance.
func SokalSneath(x, y []float64) float64 {
	var ntt, nne int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb && yb {
			ntt++
		}
		if xb != yb {
			nne++
		}
	}
	if nne == 0 {
		return 0
	}
	return float64(nne) / (0.5*float64(ntt) + float64(nne))
}

// SokalMichener computes the Sokal-Michener distance.
func SokalMichener(x, y []float64) float64 {
	n := len(x)
	var nne int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb != yb {
			nne++
		}
	}
	return float64(2*nne) / float64(n+nne)
}

// Yule computes the Yule distance.
func Yule(x, y []float64) float64 {
	var ntt, ntf, nft, nff int
	for i := range x {
		xb := x[i] != 0
		yb := y[i] != 0
		if xb && yb {
			ntt++
		} else if xb && !yb {
			ntf++
		} else if !xb && yb {
			nft++
		} else {
			nff++
		}
	}
	if ntf == 0 || nft == 0 {
		return 0
	}
	denom := ntt*nff + ntf*nft
	if denom == 0 {
		return 0
	}
	return float64(2*ntf*nft) / float64(denom)
}
