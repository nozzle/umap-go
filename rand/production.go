package rand

import (
	"math/rand/v2"
)

// Production wraps math/rand/v2.Rand as a Source.
type Production struct {
	rng *rand.Rand
}

// NewProduction creates a new production random source from a seed.
// If seed is nil, uses an auto-seeded source.
func NewProduction(seed *uint64) *Production {
	var rng *rand.Rand
	if seed != nil {
		rng = rand.New(rand.NewPCG(*seed, 0))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return &Production{rng: rng}
}

func (p *Production) Intn(n int) int {
	return p.rng.IntN(n)
}

func (p *Production) Float64() float64 {
	return p.rng.Float64()
}

func (p *Production) NormFloat64() float64 {
	return p.rng.NormFloat64()
}

func (p *Production) Perm(n int) []int {
	return p.rng.Perm(n)
}

func (p *Production) UniformFloat64(low, high float64) float64 {
	return low + p.rng.Float64()*(high-low)
}
