package umap

// curves_test.go tests FindABParams against Python reference values.

import "testing"

type findABParamsData struct {
	Results []struct {
		Spread  float64 `json:"spread"`
		MinDist float64 `json:"min_dist"`
		A       float64 `json:"a"`
		B       float64 `json:"b"`
	} `json:"results"`
}

func TestFindABParams(t *testing.T) {
	var td findABParamsData
	loadJSON(t, "find_ab_params.json", &td)

	for _, tc := range td.Results {
		t.Run("", func(t *testing.T) {
			gotA, gotB := FindABParams(tc.Spread, tc.MinDist)

			// Curve fitting may converge to slightly different values with
			// Nelder-Mead vs scipy's Levenberg-Marquardt, so use loose tolerance.
			tolA := 0.05
			tolB := 0.05

			if !approxEqual(gotA, tc.A, tolA) {
				t.Errorf("a: got %v, want %v (tol=%v)", gotA, tc.A, tolA)
			}
			if !approxEqual(gotB, tc.B, tolB) {
				t.Errorf("b: got %v, want %v (tol=%v)", gotB, tc.B, tolB)
			}
			t.Logf("spread=%.2f min_dist=%.2f → a=%.6f (want %.6f) b=%.6f (want %.6f)",
				tc.Spread, tc.MinDist, gotA, tc.A, gotB, tc.B)
		})
	}
}
