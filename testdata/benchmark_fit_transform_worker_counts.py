#!/usr/bin/env python3
"""Benchmark umap-learn FitTransform across worker counts.

Run from repository root:
  uv run --directory testdata python benchmark_fit_transform_worker_counts.py
"""

from __future__ import annotations

import json
import statistics
import time

import numpy as np
import umap


WORKER_ROWS = [
    ("w1", 1),
    ("w2", 2),
    ("w4", 4),
    ("w8", 8),
    ("auto", -1),
]

CASE = {
    "name": "n300_d10_k15_c2_e100",
    "n_samples": 300,
    "n_features": 10,
    "n_neighbors": 15,
    "n_components": 2,
    "n_epochs": 100,
    "seed": 42,
}

WARMUP_RUNS = 1
MEASURE_RUNS = 5


def run_worker(label: str, n_jobs: int, x: np.ndarray) -> dict:
    def fit_once() -> None:
        # NOTE: umap-learn overrides n_jobs to 1 when random_state is set.
        # Keep random_state unset to measure true worker scaling.
        reducer = umap.UMAP(
            n_neighbors=CASE["n_neighbors"],
            n_components=CASE["n_components"],
            min_dist=0.1,
            spread=1.0,
            n_epochs=CASE["n_epochs"],
            n_jobs=n_jobs,
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
        "label": label,
        "n_jobs": n_jobs,
        "mean_ns_per_op": mean_ns,
        "std_ns_per_op": std_ns,
        "measure_runs": MEASURE_RUNS,
        "warmup_runs": WARMUP_RUNS,
    }


def main() -> None:
    rs = np.random.RandomState(CASE["seed"])
    x = rs.rand(CASE["n_samples"], CASE["n_features"]).astype(np.float64)

    output = {
        "case": CASE,
        "rows": [run_worker(label, n_jobs, x) for label, n_jobs in WORKER_ROWS],
        "notes": "random_state intentionally unset to allow umap-learn n_jobs parallel execution",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
