#!/usr/bin/env python3
"""Run worker-count benchmark and emit README-ready table + Mermaid chart.

Usage:
  python3 testdata/benchmark_worker_counts_readme.py
  python3 testdata/benchmark_worker_counts_readme.py --update-readme
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


GO_BENCH_CMD = [
    "go",
    "test",
    "-run",
    "^$",
    "-bench",
    "^BenchmarkFitTransformWorkerCounts$",
    "-benchmem",
    "-count=1",
    ".",
]

PY_BENCH_CMDS = [
    ["uv", "run", "--directory", "testdata", "python", "benchmark_fit_transform_worker_counts.py"],
    ["python3", "testdata/benchmark_fit_transform_worker_counts.py"],
]

GO_BENCH_RE = re.compile(
    r"^BenchmarkFitTransformWorkerCounts/(?P<label>[^-\s]+)-\d+\s+\d+\s+"
    r"(?P<ns>\d+(?:\.\d+)?)\s+ns/op\s+"
    r"(?P<b>\d+)\s+B/op\s+"
    r"(?P<a>\d+)\s+allocs/op$"
)

ORDER = ["w1", "w2", "w4", "w8", "auto"]
LABEL_DISPLAY = {"w1": "1", "w2": "2", "w4": "4", "w8": "8", "auto": "auto"}


def parse_go_results(output: str) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    rows: dict[str, dict[str, float]] = {}
    meta: dict[str, str] = {}

    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith("goos: "):
            meta["goos"] = line.split(": ", 1)[1]
            continue
        if line.startswith("goarch: "):
            meta["goarch"] = line.split(": ", 1)[1]
            continue
        if line.startswith("cpu: "):
            meta["cpu"] = line.split(": ", 1)[1]
            continue

        m = GO_BENCH_RE.match(line)
        if not m:
            continue
        label = m.group("label")
        rows[label] = {
            "ns_per_op": float(m.group("ns")),
            "bytes_per_op": float(m.group("b")),
            "allocs_per_op": float(m.group("a")),
        }

    missing = [k for k in ORDER if k not in rows]
    if missing:
        raise RuntimeError(f"missing benchmark rows: {missing}")

    return rows, meta


def run_python_worker_bench() -> tuple[dict[str, float], str]:
    last_err = None
    for cmd in PY_BENCH_CMDS:
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            last_err = f"{' '.join(cmd)} failed: {exc.stderr.strip() or exc.stdout.strip()}"
            continue

        parsed = json.loads(proc.stdout)
        rows: dict[str, float] = {}
        for rec in parsed.get("rows", []):
            rows[rec["label"]] = float(rec["mean_ns_per_op"])

        missing = [k for k in ORDER if k not in rows]
        if missing:
            last_err = f"{' '.join(cmd)} missing rows: {missing}"
            continue
        return rows, " ".join(cmd)

    raise RuntimeError(last_err or "python worker benchmark command failed")


def build_snippet(go_rows: dict[str, dict[str, float]], py_rows: dict[str, float], meta: dict[str, str], py_cmd: str) -> str:
    go_baseline = go_rows["w1"]["ns_per_op"]
    py_baseline = py_rows["w1"]
    max_ns = max(
        max(v["ns_per_op"] for v in go_rows.values()),
        max(py_rows.values()),
    )
    max_axis = int(max_ns * 1.1)
    go_bars = [int(go_rows[k]["ns_per_op"]) for k in ORDER]
    py_bars = [int(py_rows[k]) for k in ORDER]

    lines: list[str] = []
    lines.append("<!-- benchmark-worker-counts:start -->")
    lines.append("")
    lines.append(
        f"_Environment: {meta.get('goos', 'unknown')}/{meta.get('goarch', 'unknown')} on {meta.get('cpu', 'unknown')}; "
        f"Go command: `{' '.join(GO_BENCH_CMD)}`; Python command: `{py_cmd}`._"
    )
    lines.append("")
    lines.append("| Workers | Go ns/op | Go speedup vs 1 | Python ns/op | Python speedup vs 1 | Python/Go | Go B/op | Go allocs/op |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for key in ORDER:
        go_rec = go_rows[key]
        py_ns = py_rows[key]
        go_speedup = go_baseline / go_rec["ns_per_op"] if go_rec["ns_per_op"] > 0 else 0.0
        py_speedup = py_baseline / py_ns if py_ns > 0 else 0.0
        py_over_go = py_ns / go_rec["ns_per_op"] if go_rec["ns_per_op"] > 0 else 0.0
        lines.append(
            f"| {LABEL_DISPLAY[key]} | {int(go_rec['ns_per_op']):,} | {go_speedup:.2f}x | {int(py_ns):,} | {py_speedup:.2f}x | {py_over_go:.2f}x | "
            f"{int(go_rec['bytes_per_op']):,} | {int(go_rec['allocs_per_op']):,} |"
        )

    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append('    title "FitTransform ns/op by worker count (Go vs Python; lower is better)"')
    lines.append('    x-axis ["1", "2", "4", "8", "auto"]')
    lines.append(f'    y-axis "ns/op" 0 --> {max_axis}')
    lines.append(f"    bar {go_bars}")
    lines.append(f"    bar {py_bars}")
    lines.append("```")
    lines.append("")
    lines.append("_Mermaid bar order: first series is Go, second series is Python (umap-learn)._")
    lines.append("")
    lines.append("<!-- benchmark-worker-counts:end -->")
    return "\n".join(lines)


def update_readme(snippet: str, readme: Path) -> None:
    content = readme.read_text()
    start = "<!-- benchmark-worker-counts:start -->"
    end = "<!-- benchmark-worker-counts:end -->"
    sidx = content.find(start)
    eidx = content.find(end)
    if sidx == -1 or eidx == -1 or eidx < sidx:
        raise RuntimeError("README benchmark markers not found")
    eidx += len(end)
    updated = content[:sidx] + snippet + content[eidx:]
    readme.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    proc = subprocess.run(GO_BENCH_CMD, check=True, capture_output=True, text=True)
    go_rows, meta = parse_go_results(proc.stdout)
    py_rows, py_cmd = run_python_worker_bench()
    snippet = build_snippet(go_rows, py_rows, meta, py_cmd)

    if args.update_readme:
        update_readme(snippet, Path("README.md"))
    else:
        print(snippet)


if __name__ == "__main__":
    main()
