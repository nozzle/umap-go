package rand

import (
	"encoding/json"
	"fmt"
	"os"
)

// RecordedData holds prerecorded random sequences from the Python reference
// implementation. Each field is a sequence of values consumed in order.
type RecordedData struct {
	Ints   []int     `json:"ints"`
	Floats []float64 `json:"floats"`
	Norms  []float64 `json:"norms"`
	Perms  [][]int   `json:"perms"`
}

// Recorded plays back prerecorded random sequences. It panics if
// more values are consumed than were recorded, indicating a divergence
// between Go and Python call order.
type Recorded struct {
	data RecordedData
	iIdx int
	fIdx int
	nIdx int
	pIdx int
}

// NewRecorded creates a Recorded source from a RecordedData struct.
func NewRecorded(data RecordedData) *Recorded {
	return &Recorded{data: data}
}

// NewRecordedFromFile loads a Recorded source from a JSON file.
func NewRecordedFromFile(path string) (*Recorded, error) {
	f, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("rand: reading recorded data: %w", err)
	}
	var data RecordedData
	if err := json.Unmarshal(f, &data); err != nil {
		return nil, fmt.Errorf("rand: parsing recorded data: %w", err)
	}
	return NewRecorded(data), nil
}

func (r *Recorded) Intn(n int) int {
	if r.iIdx >= len(r.data.Ints) {
		panic(fmt.Sprintf("rand.Recorded: exhausted int sequence at index %d (had %d values)", r.iIdx, len(r.data.Ints)))
	}
	v := r.data.Ints[r.iIdx]
	r.iIdx++
	// The recorded value is the raw output from Python; take mod n to match
	// the expected range, same as the caller would do.
	if n > 0 {
		result := v % n
		if result < 0 {
			result += n
		}
		return result
	}
	return v
}

func (r *Recorded) Float64() float64 {
	if r.fIdx >= len(r.data.Floats) {
		panic(fmt.Sprintf("rand.Recorded: exhausted float sequence at index %d (had %d values)", r.fIdx, len(r.data.Floats)))
	}
	v := r.data.Floats[r.fIdx]
	r.fIdx++
	return v
}

func (r *Recorded) NormFloat64() float64 {
	if r.nIdx >= len(r.data.Norms) {
		panic(fmt.Sprintf("rand.Recorded: exhausted norm sequence at index %d (had %d values)", r.nIdx, len(r.data.Norms)))
	}
	v := r.data.Norms[r.nIdx]
	r.nIdx++
	return v
}

func (r *Recorded) Perm(n int) []int {
	if r.pIdx >= len(r.data.Perms) {
		panic(fmt.Sprintf("rand.Recorded: exhausted perm sequence at index %d (had %d values)", r.pIdx, len(r.data.Perms)))
	}
	v := r.data.Perms[r.pIdx]
	r.pIdx++
	if len(v) != n {
		panic(fmt.Sprintf("rand.Recorded: perm length mismatch: requested %d, recorded %d", n, len(v)))
	}
	return v
}

func (r *Recorded) UniformFloat64(low, high float64) float64 {
	return low + r.Float64()*(high-low)
}

// Remaining returns the number of unconsumed values in each sequence.
func (r *Recorded) Remaining() (ints, floats, norms, perms int) {
	return len(r.data.Ints) - r.iIdx,
		len(r.data.Floats) - r.fIdx,
		len(r.data.Norms) - r.nIdx,
		len(r.data.Perms) - r.pIdx
}
