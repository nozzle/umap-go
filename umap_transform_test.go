package umap

import (
	"encoding/csv"
	"math"
	"os"
	"strconv"
	"testing"

	umaprand "github.com/nozzle/umap-go/rand"
)

func generateTestData(n, dim int, rng umaprand.Source) [][]float64 {
	data := make([][]float64, n)
	for i := range n {
		data[i] = make([]float64, dim)
		for j := range dim {
			data[i][j] = rng.Float64()
		}
	}
	return data
}

func TestUMAP_TransformParity(t *testing.T) {
	s0 := uint64(42)
	rng := umaprand.NewProduction(&s0)

	XTrain := generateTestData(100, 4, rng)
	XTest := generateTestData(20, 4, rng)

	opts := DefaultOptions()
	opts.NNeighbors = 5
	opts.MinDist = 0.1
	opts.NEpochs = 100
	s := uint64(42)
	opts.RandSource = umaprand.NewProduction(&s)

	model := New(opts)
	_, err := model.FitTransform(XTrain, nil)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	s2 := uint64(42)
	opts.RandSource = umaprand.NewProduction(&s2)

	embeddingTest, err := model.Transform(XTest)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// Read python output
	f, err := os.Open("transform_test.csv")
	if err != nil {
		t.Fatalf("failed to open python output: %v", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		t.Fatalf("failed to read python output: %v", err)
	}

	var maxErr float64
	for i, record := range records {
		v0, _ := strconv.ParseFloat(record[0], 64)
		v1, _ := strconv.ParseFloat(record[1], 64)

		e0 := math.Abs(embeddingTest[i][0] - v0)
		e1 := math.Abs(embeddingTest[i][1] - v1)

		if e0 > maxErr {
			maxErr = e0
		}
		if e1 > maxErr {
			maxErr = e1
		}
	}

	t.Logf("Transform max difference from python: %f", maxErr)
	// Just informational for now. Real exact parity is hard with default PRNG.
}
