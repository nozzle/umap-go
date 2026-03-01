#!/usr/bin/env python3
"""Benchmark Python umap-learn FitTransform for comparison with umap-go.

Run from repository root:
  uv run python testdata/benchmark_fit_transform.py
"""

from __future__ import annotations

import json
import statistics
import time

import numpy as np
import umap


CASES = [
    {
        "name": "n300_d10_k15_c2_e100",
        "n_samples": 300,
        "n_features": 10,
        "n_neighbors": 15,
        "n_components": 2,
        "n_epochs": 100,
        "seed": 42,
    },
    {
        "name": "n600_d20_k15_c2_e100",
        "n_samples": 600,
        "n_features": 20,
        "n_neighbors": 15,
        "n_components": 2,
        "n_epochs": 100,
        "seed": 42,
    },
]

WARMUP_RUNS = 1
MEASURE_RUNS = 5


def run_case(case: dict) -> dict:
    rs = np.random.RandomState(case["seed"])
    x = rs.rand(case["n_samples"], case["n_features"]).astype(np.float64)

    def fit_once() -> None:
        reducer = umap.UMAP(
            n_neighbors=case["n_neighbors"],
            n_components=case["n_components"],
            min_dist=0.1,
            spread=1.0,
            random_state=int(case["seed"]),
            n_epochs=case["n_epochs"],
        )
        reducer.fit_transform(x)

    for _ in range(WARMUP_RUNS):
        fit_once()

    timings_ns: list[int] = []
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter_ns()
        fit_once()
        t1 = time.perf_counter_ns()
        timings_ns.append(t1 - t0)

    mean_ns = statistics.fmean(timings_ns)
    std_ns = statistics.pstdev(timings_ns) if len(timings_ns) > 1 else 0.0

    return {
        "name": case["name"],
        "n_samples": case["n_samples"],
        "n_features": case["n_features"],
        "n_neighbors": case["n_neighbors"],
        "n_components": case["n_components"],
        "n_epochs": case["n_epochs"],
        "seed": case["seed"],
        "warmup_runs": WARMUP_RUNS,
        "measure_runs": MEASURE_RUNS,
        "mean_ns_per_op": mean_ns,
        "std_ns_per_op": std_ns,
    }


def main() -> None:
    output = {"cases": [run_case(case) for case in CASES]}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
