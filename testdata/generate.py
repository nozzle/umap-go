#!/usr/bin/env python3
"""Generate test data for umap-go by recording Python UMAP intermediate outputs.

Run with:  uv run python generate.py

This records each algorithmic stage as a separate JSON file under
testdata/<dataset>/<stage>.json so the Go test suite can do
byte-for-byte (or tolerance-based) comparisons per stage.

Pinned to umap-learn==0.5.11.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from numpy.random import RandomState
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.datasets import load_iris, make_blobs

import umap
import umap.distances as dist
import umap.umap_ as umap_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_list(arr):
    """Convert numpy arrays to JSON-serializable lists."""
    if isinstance(arr, np.ndarray):
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=-99999.99, posinf=99999.99, neginf=-99999.99)
        return arr.tolist()
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=-99999.99, posinf=99999.99, neginf=-99999.99)
        return arr.tolist()
    return arr


def save_json(path: str, obj: dict):
    """Write obj to path as indented JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)
    print(f"  wrote {path}")


def sparse_to_dict(m):
    """Convert a scipy sparse matrix to a JSON-friendly dict."""
    coo = coo_matrix(m)
    return {
        "shape": list(coo.shape),
        "row": coo.row.tolist(),
        "col": coo.col.tolist(),
        "data": coo.data.tolist(),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def load_datasets() -> dict[str, dict]:
    """Return dict of dataset_name -> {X, y, description}."""
    datasets = {}

    # 1. Iris (primary test dataset, small, well-known)
    iris = load_iris()
    datasets["iris"] = {
        "X": iris.data.astype(np.float64),
        "y": iris.target.astype(np.float64),
        "description": "Iris dataset (150x4)",
    }

    # 2. Synthetic blobs (simple clustering)
    X_blobs, y_blobs = make_blobs(
        n_samples=200, n_features=10, centers=5, random_state=42
    )
    datasets["blobs"] = {
        "X": X_blobs.astype(np.float64),
        "y": y_blobs.astype(np.float64),
        "description": "Synthetic blobs (200x10, 5 centers)",
    }

    # 3. Small dataset (for brute-force path)
    X_small, y_small = make_blobs(n_samples=30, n_features=5, centers=3, random_state=7)
    datasets["small"] = {
        "X": X_small.astype(np.float64),
        "y": y_small.astype(np.float64),
        "description": "Small blobs (30x5, for brute-force kNN)",
    }

    # Data for specific distance metrics
    rs = RandomState(42)
    datasets["binary"] = {
        "X": rs.randint(2, size=(20, 10)).astype(np.float64),
        "description": "Binary data for jaccard, hamming, etc.",
    }

    probs = rs.rand(20, 5).astype(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    datasets["probs"] = {
        "X": probs,
        "description": "Probability distributions for hellinger, etc.",
    }

    datasets["2d"] = {
        "X": rs.uniform(-np.pi / 2, np.pi / 2, size=(20, 2)).astype(np.float64),
        "description": "2D data for haversine",
    }

    datasets["sparse"] = {
        "X": (rs.rand(20, 10) > 0.7).astype(np.float64),
        "description": "Sparse data",
    }

    return datasets


# ---------------------------------------------------------------------------
# Stage 1: find_ab_params
# ---------------------------------------------------------------------------


def record_find_ab_params(outdir: str):
    """Record the a,b curve-fitting result for several (spread, min_dist) pairs."""
    print("Stage 1: find_ab_params")
    results = []
    for spread, min_dist in [
        (1.0, 0.1),
        (1.0, 0.25),
        (1.0, 0.5),
        (1.5, 0.1),
        (2.0, 0.5),
    ]:
        a, b = umap_mod.find_ab_params(spread, min_dist)
        results.append(
            {
                "spread": spread,
                "min_dist": min_dist,
                "a": float(a),
                "b": float(b),
            }
        )
    save_json(os.path.join(outdir, "find_ab_params.json"), {"results": results})


# ---------------------------------------------------------------------------
# Stage 2: Pairwise distances (small dataset, brute-force)
# ---------------------------------------------------------------------------


def record_pairwise_distances(outdir: str, datasets: dict):
    """Record brute-force pairwise distances for the small dataset."""
    print("Stage 2: pairwise_distances")

    # 1. Standard metrics
    metrics = [
        "euclidean",
        "l2",
        "manhattan",
        "taxicab",
        "l1",
        "chebyshev",
        "linfinity",
        "linfty",
        "linf",
        "canberra",
        "braycurtis",
        "cosine",
        "correlation",
        "poincare",
        "hellinger",
        "haversine",
        "ll_dirichlet",
        "symmetric_kl",
        "jaccard",
        "dice",
        "hamming",
        "matching",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalsneath",
        "sokalmichener",
        "yule",
    ]

    for metric_name in metrics:
        if metric_name in [
            "jaccard",
            "dice",
            "hamming",
            "matching",
            "kulsinski",
            "rogerstanimoto",
            "russellrao",
            "sokalsneath",
            "sokalmichener",
            "yule",
        ]:
            ds = "binary"
        elif metric_name in ["hellinger", "ll_dirichlet", "symmetric_kl"]:
            ds = "probs"
        elif metric_name == "haversine":
            ds = "2d"
        elif metric_name == "poincare":
            # Poincare needs values strictly inside the unit disk
            X_poincare = datasets["small"]["X"].copy()
            norms = np.linalg.norm(X_poincare, axis=1, keepdims=True)
            X_poincare = X_poincare / (norms + 1.0)  # Ensure norm < 1
            datasets["poincare"] = {"X": X_poincare * 0.9}
            ds = "poincare"
        else:
            ds = "small"

        X = datasets[ds]["X"]
        n = X.shape[0]
        fn = dist.named_distances[metric_name]
        D = np.zeros((n, n))

        try:
            for i in range(n):
                for j in range(n):
                    D[i, j] = fn(X[i], X[j])

            save_json(
                os.path.join(outdir, "pairwise_distances", f"{metric_name}.json"),
                {
                    "metric": metric_name,
                    "X": to_list(X),
                    "distances": to_list(D),
                },
            )
        except Exception as e:
            print(f"  Failed metric {metric_name}: {e}")

    # 2. Gradient metrics
    print("Stage 2.1: pairwise_gradients")
    grad_metrics = [
        "euclidean",
        "l2",
        "manhattan",
        "taxicab",
        "l1",
        "chebyshev",
        "linfinity",
        "linfty",
        "linf",
        "cosine",
        "correlation",
        "canberra",
        "braycurtis",
        "hellinger",
        "haversine",
        "hyperboloid",
    ]

    for metric_name in grad_metrics:
        if metric_name == "hellinger":
            ds = "probs"
        elif metric_name == "haversine":
            ds = "2d"
        elif metric_name == "hyperboloid":
            # Hyperboloid needs points in hyperbolic space (z > 0, -t0^2 + x1^2 +... = -1)
            # UMAP's hyperboloid dist treats the first coordinate as the t0.
            # We'll generate simple valid points by using a standard dataset and augmenting it.
            # Actually, let's just create points where x0 = sqrt(1 + sum(xi^2)).
            X_hyp = datasets["small"]["X"].copy()
            x0 = np.sqrt(1.0 + np.sum(X_hyp**2, axis=1)).reshape(-1, 1)
            X_hyp = np.hstack([x0, X_hyp])
            datasets["hyperboloid"] = {"X": X_hyp}
            ds = "hyperboloid"
        else:
            ds = "small"

        X = datasets[ds]["X"]
        n = min(10, X.shape[0])  # keep gradients smaller
        X = X[:n]
        fn = dist.named_distances_with_gradients[metric_name]

        D = np.zeros((n, n))
        grads = np.zeros((n, n, X.shape[1]))

        try:
            for i in range(n):
                for j in range(n):
                    d, grad = fn(X[i], X[j])
                    D[i, j] = d
                    grads[i, j] = grad

            save_json(
                os.path.join(outdir, "pairwise_gradients", f"{metric_name}.json"),
                {
                    "metric": metric_name,
                    "X": to_list(X),
                    "distances": to_list(D),
                    "gradients": to_list(grads),
                },
            )
        except Exception as e:
            print(f"  Failed grad metric {metric_name}: {e}")

    # 3. Sparse metrics
    print("Stage 2.2: sparse_distances")
    import umap.sparse as sparse

    sparse_metrics = [
        "euclidean",
        "manhattan",
        "taxicab",
        "l1",
        "chebyshev",
        "linfinity",
        "linfty",
        "linf",
        "canberra",
        "braycurtis",
        "cosine",
        "correlation",
        "hellinger",
        "ll_dirichlet",
        "jaccard",
        "dice",
        "hamming",
        "matching",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalsneath",
        "sokalmichener",
    ]

    for metric_name in sparse_metrics:
        if metric_name in [
            "jaccard",
            "dice",
            "hamming",
            "matching",
            "kulsinski",
            "rogerstanimoto",
            "russellrao",
            "sokalsneath",
            "sokalmichener",
        ]:
            ds = "binary"
        elif metric_name in ["hellinger", "ll_dirichlet"]:
            ds = "probs"
        else:
            ds = "sparse"

        X = datasets[ds]["X"]
        X_sparse = csr_matrix(X)
        n = X_sparse.shape[0]

        fn = sparse.sparse_named_distances[metric_name]
        D = np.zeros((n, n))

        try:
            for i in range(n):
                for j in range(n):
                    ind1 = X_sparse.indices[X_sparse.indptr[i] : X_sparse.indptr[i + 1]]
                    data1 = X_sparse.data[X_sparse.indptr[i] : X_sparse.indptr[i + 1]]
                    ind2 = X_sparse.indices[X_sparse.indptr[j] : X_sparse.indptr[j + 1]]
                    data2 = X_sparse.data[X_sparse.indptr[j] : X_sparse.indptr[j + 1]]

                    if metric_name in [
                        "hamming",
                        "matching",
                        "kulsinski",
                        "rogerstanimoto",
                        "russellrao",
                        "sokalmichener",
                        "correlation",
                    ]:
                        D[i, j] = fn(ind1, data1, ind2, data2, X_sparse.shape[1])
                    else:
                        D[i, j] = fn(ind1, data1, ind2, data2)

            save_json(
                os.path.join(outdir, "sparse_distances", f"{metric_name}.json"),
                {
                    "metric": metric_name,
                    "X_shape": list(X_sparse.shape),
                    "X_indptr": to_list(X_sparse.indptr),
                    "X_indices": to_list(X_sparse.indices),
                    "X_data": to_list(X_sparse.data),
                    "distances": to_list(D),
                },
            )
        except Exception as e:
            print(f"  Failed sparse metric {metric_name}: {e}")


# ---------------------------------------------------------------------------
# Stage 3: Nearest neighbors (brute-force fast_knn_indices)
# ---------------------------------------------------------------------------


def record_knn_nndescent(outdir: str, datasets: dict):
    """Record NNDescent kNN for a dataset > 4096 samples."""
    print("Stage 3.1: knn_nndescent")
    # Generate 4100 random points so it trips the > 4096 logic
    rs = RandomState(42)
    X = rs.rand(4100, 10)
    n_neighbors = 15

    knn_indices, knn_dists, _ = umap_mod.nearest_neighbors(
        X,
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds={},
        angular=False,
        random_state=RandomState(42),
        n_jobs=1,  # Force sequential to keep PRNG deterministic!
    )

    save_json(
        os.path.join(outdir, "knn_nndescent", "euclidean.json"),
        {
            "X": to_list(X),
            "n_neighbors": n_neighbors,
            "metric": "euclidean",
            "knn_indices": to_list(knn_indices),
            "knn_dists": to_list(knn_dists),
        },
    )


def record_knn_brute(outdir: str, datasets: dict):
    """Record brute-force kNN for the small dataset."""
    print("Stage 3: knn_brute_force")
    X = datasets["small"]["X"]
    n_neighbors = 10

    # fast_knn_indices is used when n_samples < 4096
    knn_indices, knn_dists, _ = umap_mod.nearest_neighbors(
        X,
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds={},
        angular=False,
        random_state=RandomState(42),
    )

    save_json(
        os.path.join(outdir, "knn_brute", "euclidean.json"),
        {
            "X": to_list(X),
            "n_neighbors": n_neighbors,
            "metric": "euclidean",
            "knn_indices": to_list(knn_indices),
            "knn_dists": to_list(knn_dists),
        },
    )


# ---------------------------------------------------------------------------
# Stage 4: smooth_knn_dist
# ---------------------------------------------------------------------------


def record_smooth_knn_dist(outdir: str, datasets: dict):
    """Record smooth_knn_dist results."""
    print("Stage 4: smooth_knn_dist")

    for ds_name in ["iris", "small"]:
        X = datasets[ds_name]["X"]
        n_neighbors = 15

        knn_indices, knn_dists, _ = umap_mod.nearest_neighbors(
            X,
            n_neighbors=n_neighbors,
            metric="euclidean",
            metric_kwds={},
            angular=False,
            random_state=RandomState(42),
        )

        sigmas, rhos = umap_mod.smooth_knn_dist(
            knn_dists, float(n_neighbors), local_connectivity=1.0
        )

        save_json(
            os.path.join(outdir, "smooth_knn_dist", f"{ds_name}.json"),
            {
                "knn_dists": to_list(knn_dists),
                "n_neighbors": n_neighbors,
                "local_connectivity": 1.0,
                "sigmas": to_list(sigmas),
                "rhos": to_list(rhos),
            },
        )


# ---------------------------------------------------------------------------
# Stage 5: compute_membership_strengths → fuzzy_simplicial_set
# ---------------------------------------------------------------------------


def record_fuzzy_simplicial_set(outdir: str, datasets: dict):
    """Record the full fuzzy_simplicial_set output (the UMAP graph)."""
    print("Stage 5: fuzzy_simplicial_set")

    for ds_name in ["iris", "small"]:
        X = datasets[ds_name]["X"]
        n_neighbors = 15
        rs = RandomState(42)

        graph, sigmas, rhos = umap_mod.fuzzy_simplicial_set(
            X,
            n_neighbors=n_neighbors,
            random_state=rs,
            metric="euclidean",
        )

        save_json(
            os.path.join(outdir, "fuzzy_simplicial_set", f"{ds_name}.json"),
            {
                "n_neighbors": n_neighbors,
                "metric": "euclidean",
                "graph": sparse_to_dict(graph),
                "sigmas": to_list(sigmas),
                "rhos": to_list(rhos),
            },
        )


# ---------------------------------------------------------------------------
# Stage 6: make_epochs_per_sample
# ---------------------------------------------------------------------------


def record_epochs_per_sample(outdir: str, datasets: dict):
    """Record epochs_per_sample computation."""
    print("Stage 6: make_epochs_per_sample")

    for ds_name in ["iris"]:
        X = datasets[ds_name]["X"]
        n_neighbors = 15
        rs = RandomState(42)

        graph, _, _ = umap_mod.fuzzy_simplicial_set(
            X,
            n_neighbors=n_neighbors,
            random_state=rs,
            metric="euclidean",
        )

        n_epochs = 200
        graph_data = graph.tocoo()
        graph_data.sum_duplicates()

        eps = umap_mod.make_epochs_per_sample(graph_data.data, n_epochs)

        save_json(
            os.path.join(outdir, "epochs_per_sample", f"{ds_name}.json"),
            {
                "n_epochs": n_epochs,
                "graph_data": to_list(graph_data.data),
                "epochs_per_sample": to_list(eps),
            },
        )


# ---------------------------------------------------------------------------
# Stage 7: Spectral layout initialization
# ---------------------------------------------------------------------------


def record_spectral_layout(outdir: str, datasets: dict):
    """Record spectral initialization.

    Note: eigenvectors may be sign-flipped relative to Go's decomposition.
    We test subspace equivalence, not exact values.

    We also save the fuzzy simplicial set graph so Go can run SpectralLayout
    on the exact same graph rather than rebuilding it (which would differ
    due to kNN implementation differences).
    """
    print("Stage 7: spectral_layout")
    import umap.spectral as spectral

    for ds_name in ["iris"]:
        X = datasets[ds_name]["X"]
        n_neighbors = 15
        rs = RandomState(42)

        graph, _, _ = umap_mod.fuzzy_simplicial_set(
            X,
            n_neighbors=n_neighbors,
            random_state=rs,
            metric="euclidean",
        )

        n_components = 2
        init = spectral.spectral_layout(
            X,
            graph,
            n_components,
            rs,
        )

        # Convert graph to CSR for Go consumption
        graph_csr = csr_matrix(graph)
        graph_csr.sort_indices()

        save_json(
            os.path.join(outdir, "spectral_layout", f"{ds_name}.json"),
            {
                "n_components": n_components,
                "init": to_list(init),
                "graph": {
                    "shape": list(graph_csr.shape),
                    "indptr": graph_csr.indptr.tolist(),
                    "indices": graph_csr.indices.tolist(),
                    "data": graph_csr.data.tolist(),
                },
            },
        )


# ---------------------------------------------------------------------------
# Stage 8: Full UMAP pipeline (end-to-end)
# ---------------------------------------------------------------------------


def record_full_umap(outdir: str, datasets: dict):
    """Record end-to-end UMAP output for regression tests."""
    print("Stage 8: full_umap")

    for ds_name in ["iris", "blobs"]:
        X = datasets[ds_name]["X"]
        y = datasets[ds_name]["y"]

        # Unsupervised
        reducer = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            spread=1.0,
            random_state=42,
            n_epochs=200,
        )
        embedding = reducer.fit_transform(X)

        save_json(
            os.path.join(outdir, "full_umap", f"{ds_name}_unsupervised.json"),
            {
                "n_neighbors": 15,
                "n_components": 2,
                "min_dist": 0.1,
                "spread": 1.0,
                "random_state": 42,
                "n_epochs": 200,
                "X": to_list(X),
                "embedding": to_list(embedding),
                "a": float(reducer._a),
                "b": float(reducer._b),
            },
        )

        # Supervised
        reducer_sup = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            spread=1.0,
            random_state=42,
            n_epochs=200,
        )
        embedding_sup = reducer_sup.fit_transform(X, y)

        save_json(
            os.path.join(outdir, "full_umap", f"{ds_name}_supervised.json"),
            {
                "n_neighbors": 15,
                "n_components": 2,
                "min_dist": 0.1,
                "spread": 1.0,
                "random_state": 42,
                "n_epochs": 200,
                "X": to_list(X),
                "y": to_list(y),
                "embedding": to_list(embedding_sup),
                "a": float(reducer_sup._a),
                "b": float(reducer_sup._b),
            },
        )


# ---------------------------------------------------------------------------
# Stage 9: Transform (project new points)
# ---------------------------------------------------------------------------


def record_transform(outdir: str, datasets: dict):
    """Record transform of held-out points."""
    print("Stage 9: transform")

    X = datasets["iris"]["X"]

    # Use first 120 to train, last 30 to transform
    X_train = X[:120]
    X_test = X[120:]

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        random_state=42,
        n_epochs=200,
    )
    reducer.fit(X_train)
    test_embedding = reducer.transform(X_test)

    save_json(
        os.path.join(outdir, "transform", "iris.json"),
        {
            "X_train": to_list(X_train),
            "X_test": to_list(X_test),
            "test_embedding": to_list(test_embedding),
            "train_embedding": to_list(reducer.embedding_),
        },
    )


# ---------------------------------------------------------------------------
# Stage 10: SGD layout optimization (record init + final for small dataset)
# ---------------------------------------------------------------------------


def record_sgd_layout(outdir: str, datasets: dict):
    """Record SGD layout optimization with known init for small dataset."""
    print("Stage 10: sgd_layout")
    import umap.layouts as layouts

    X = datasets["small"]["X"]
    n_neighbors = 10
    rs = RandomState(42)

    graph, _, _ = umap_mod.fuzzy_simplicial_set(
        X,
        n_neighbors=n_neighbors,
        random_state=rs,
        metric="euclidean",
    )

    n_epochs = 200
    a, b = umap_mod.find_ab_params(1.0, 0.1)

    # Use random init so we have a deterministic starting point
    rng = RandomState(42)
    n = X.shape[0]
    init = rng.uniform(low=-10.0, high=10.0, size=(n, 2)).astype(np.float32)

    graph_coo = graph.tocoo()
    graph_coo.sum_duplicates()

    eps = umap_mod.make_epochs_per_sample(graph_coo.data, n_epochs)

    head = graph_coo.row
    tail = graph_coo.col

    # The rng_state for optimize_layout_euclidean is a (3,) int64 array
    # representing a single Tausworthe PRNG state. The function internally
    # broadcasts it to (n_vertices, 3) by adding per-sample offsets.
    sgd_rng = RandomState(42)
    rng_state = sgd_rng.randint(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=3
    ).astype(np.int64)

    embedding = layouts.optimize_layout_euclidean(
        init.copy(),
        init.copy(),
        head,
        tail,
        n_epochs,
        n,
        eps,
        a,
        b,
        rng_state,
        gamma=1.0,
        initial_alpha=1.0,
        negative_sample_rate=5,
        parallel=False,
        verbose=False,
    )

    save_json(
        os.path.join(outdir, "sgd_layout", "small.json"),
        {
            "init": to_list(init),
            "head": to_list(head),
            "tail": to_list(tail),
            "n_epochs": n_epochs,
            "a": float(a),
            "b": float(b),
            "gamma": 1.0,
            "initial_alpha": 1.0,
            "negative_sample_rate": 5,
            "epochs_per_sample": to_list(eps),
            "rng_state": to_list(rng_state),
            "embedding": to_list(embedding),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    print(f"Output directory: {outdir}")
    print(f"UMAP version: {umap.__version__}")

    datasets = load_datasets()

    # Save raw datasets
    for name, ds in datasets.items():
        save_json(
            os.path.join(outdir, "datasets", f"{name}.json"),
            {
                "description": ds.get("description", ""),
                "X": to_list(ds["X"]),
                "y": to_list(ds.get("y", [])),
                "shape": list(ds["X"].shape),
            },
        )

    # Record all stages
    record_find_ab_params(outdir)
    record_pairwise_distances(outdir, datasets)
    record_knn_brute(outdir, datasets)
    record_knn_nndescent(outdir, datasets)
    record_smooth_knn_dist(outdir, datasets)
    record_fuzzy_simplicial_set(outdir, datasets)
    record_epochs_per_sample(outdir, datasets)
    record_spectral_layout(outdir, datasets)
    record_full_umap(outdir, datasets)
    record_transform(outdir, datasets)
    record_sgd_layout(outdir, datasets)

    print("\nDone! All test data generated.")


if __name__ == "__main__":
    main()
