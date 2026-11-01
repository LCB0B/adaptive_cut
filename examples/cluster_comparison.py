"""The adaptive cut on scikit-learn's toy datasets, next to scikit-learn's own methods.

The datasets, the methods and their parameters are those of scikit-learn's example
"Comparing different clustering algorithms on toy datasets" (``plot_cluster_comparison.py``,
BSD-3-Clause), where every method gets parameters tuned for each dataset, including the true
number of clusters. The adaptive cut is never told the number of clusters, and uses one setting
for every dataset: a single-linkage tree, the partition scored by its modularity on the graph of
the 20 nearest neighbours of each point, at resolution 0.1 (``modularity_scores``). It finds
connected shapes of any form, and merges blobs that touch.

``python examples/cluster_comparison.py`` prints its clusters on every dataset (needs
scikit-learn).
"""

import time
import warnings

import numpy as np

import adaptive_cut as ac

N_NEIGHBORS, RESOLUTION = 20, 0.1


def knn_modularity(Z, X):
    """Modularity of every tree node on the symmetric k-nearest-neighbour graph of ``X``."""
    from sklearn.neighbors import kneighbors_graph

    A = kneighbors_graph(X, N_NEIGHBORS, include_self=False)
    return ac.modularity_scores(Z, ((A + A.T) > 0).astype(float), resolution=RESOLUTION)


ADAPTIVE = "Adaptive cut"


def adaptive(seed=0):
    """The adaptive cut of the single-linkage tree under the k-nearest-neighbour modularity."""
    return ac.AdaptiveClustering(linkage="single", objective=knn_modularity, random_state=seed)


def datasets():
    """scikit-learn's six toy datasets, standardised, each with the example's parameters."""
    from sklearn import datasets as d
    from sklearn.preprocessing import StandardScaler

    n, seed = 500, 30
    X, y = d.make_blobs(n_samples=n, random_state=170)
    data = [
        ("noisy circles", d.make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=seed)),
        ("noisy moons", d.make_moons(n_samples=n, noise=0.05, random_state=seed)),
        ("varied", d.make_blobs(n_samples=n, cluster_std=[1.0, 2.5, 0.5], random_state=170)),
        ("anisotropic", (X @ [[0.6, -0.6], [-0.4, 0.8]], y)),
        ("blobs", d.make_blobs(n_samples=n, random_state=seed)),
        ("no structure", (np.random.RandomState(seed).rand(n, 2), None)),
    ]
    params = [
        dict(damping=0.77, preference=-240, quantile=0.2, n_clusters=2, min_samples=7, xi=0.08),
        dict(damping=0.75, preference=-220, n_clusters=2, min_samples=7, xi=0.1),
        dict(eps=0.18, n_neighbors=2, min_samples=7, xi=0.01, min_cluster_size=0.2),
        dict(eps=0.15, n_neighbors=2, min_samples=7, xi=0.1, min_cluster_size=0.2),
        dict(min_samples=7, xi=0.1, min_cluster_size=0.2),
        dict(),
    ]
    return [
        (name, StandardScaler().fit_transform(X), y, p) for (name, (X, y)), p in zip(data, params)
    ]


def sklearn_methods(X, params):
    """scikit-learn's methods with the example's parameters for this dataset."""
    from sklearn import cluster, mixture
    from sklearn.neighbors import kneighbors_graph

    p = dict(
        quantile=0.3,
        eps=0.3,
        damping=0.9,
        preference=-200,
        n_neighbors=3,
        n_clusters=3,
        min_samples=7,
        xi=0.05,
        min_cluster_size=0.1,
        allow_single_cluster=True,
        hdbscan_min_cluster_size=15,
        hdbscan_min_samples=3,
        random_state=42,
    )
    p.update(params)
    connectivity = kneighbors_graph(X, n_neighbors=p["n_neighbors"], include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    k, seed = p["n_clusters"], p["random_state"]
    return {
        "MiniBatch\nKMeans": cluster.MiniBatchKMeans(n_clusters=k, random_state=seed),
        "Affinity\nPropagation": cluster.AffinityPropagation(
            damping=p["damping"], preference=p["preference"], random_state=seed
        ),
        "MeanShift": cluster.MeanShift(
            bandwidth=cluster.estimate_bandwidth(X, quantile=p["quantile"]), bin_seeding=True
        ),
        "Spectral\nClustering": cluster.SpectralClustering(
            n_clusters=k, eigen_solver="arpack", affinity="nearest_neighbors", random_state=seed
        ),
        "Ward": cluster.AgglomerativeClustering(
            n_clusters=k, linkage="ward", connectivity=connectivity
        ),
        "Agglomerative\nClustering": cluster.AgglomerativeClustering(
            linkage="average", metric="cityblock", n_clusters=k, connectivity=connectivity
        ),
        "DBSCAN": cluster.DBSCAN(eps=p["eps"]),
        "HDBSCAN": cluster.HDBSCAN(
            min_samples=p["hdbscan_min_samples"],
            min_cluster_size=p["hdbscan_min_cluster_size"],
            allow_single_cluster=p["allow_single_cluster"],
            copy=True,
        ),
        "OPTICS": cluster.OPTICS(
            min_samples=p["min_samples"], xi=p["xi"], min_cluster_size=p["min_cluster_size"]
        ),
        "BIRCH": cluster.Birch(n_clusters=k),
        "Gaussian\nMixture": mixture.GaussianMixture(
            n_components=k, covariance_type="full", random_state=seed
        ),
    }


def adaptive_table(seed=0):
    """The adaptive cut's clusters on every dataset, as rows of the README table."""
    from sklearn.metrics import adjusted_rand_score

    rows = []
    for name, X, y, _ in datasets():
        labels = adaptive(seed).fit(X).labels_
        k = len(np.unique(labels))
        cell = f"{k} cluster" + ("s" if k > 1 else "")
        if y is not None:
            cell += f", ARI {adjusted_rand_score(y, labels):.2f}"
        rows.append(f"| {name} | {cell} |")
    return rows


def run(seed=0):
    """Every method on every dataset: labels and fit time (seconds) of each."""
    data = datasets()
    # compile the numba sampler before anything is timed
    ac.AdaptiveClustering(random_state=seed).fit(data[0][1][:50])
    rows = []
    for name, X, y, params in data:
        methods = sklearn_methods(X, params) | {ADAPTIVE: adaptive(seed)}
        results = {}
        for method, model in methods.items():
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # graph warnings, as in the example
                model.fit(X)
            seconds = time.perf_counter() - t0
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)
            results[method] = (np.asarray(labels, dtype=int), seconds)
        rows.append((name, X, y, results))
    return rows


if __name__ == "__main__":
    print("\n".join(adaptive_table()))
