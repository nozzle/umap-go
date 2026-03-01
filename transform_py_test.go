package umap

import (
	"encoding/csv"
	"os"
	"strconv"
	"fmt"
)

// Read CSV float matrix
func readCSV(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	res := make([][]float64, len(records))
	for i, row := range records {
		res[i] = make([]float64, len(row))
		for j, val := range row {
			res[i][j], _ = strconv.ParseFloat(val, 64)
		}
	}
	return res, nil
}

// Write CSV float matrix
func writeCSV(path string, data [][]float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	for _, row := range data {
		record := make([]string, len(row))
		for j, val := range row {
			record[j] = fmt.Sprintf("%.8f", val)
		}
		w.Write(record)
	}
	w.Flush()
	return w.Error()
}
