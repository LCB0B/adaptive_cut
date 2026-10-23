"""How much does the adaptive cut improve standard hierarchical clustering?

For every dataset and every dendrogram method, the same unsupervised objective
(a per-cluster Gaussian BIC) is maximised by

* the best single-level ("horizontal") cut of the dendrogram, and
* the adaptive cut (``adaptive_cut.adaptive_cut``) with two settings:
  "default" (``T="auto"``, starting from the single-level cut, default number
  of steps) and "hot" (100 times the automatic temperature, 100,000 steps).

Both are scored against the ground-truth labels with ARI and AMI. As a
reference, the single cut at the true number of clusters (scipy ``fcluster``
with ``maxclust=k_true``) is also reported; it uses information an
unsupervised method does not have.

Run with ``python benchmarks/clustering.py``; results go to
``benchmarks/results/clustering.csv``. Needs scikit-learn.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker process

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

import adaptive_cut as ac

HERE = Path(__file__).resolve().parent

METHODS = ["single", "complete", "average", "weighted", "centroid", "median", "ward", "hdbscan"]
SEEDS = range(10)
N_SYNTH = 1000


def synthetic(name, seed):
    from sklearn import datasets

    rng = np.random.default_rng(seed)
    n = N_SYNTH
    if name == "circles":
        return datasets.make_circles(n, factor=0.5, noise=0.05, random_state=seed)
    if name == "moons":
        return datasets.make_moons(n, noise=0.05, random_state=seed)
    if name == "blobs":
        return datasets.make_blobs(n, random_state=seed)
    if name == "varied":
        return datasets.make_blobs(n, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    if name == "aniso":
        X, y = datasets.make_blobs(n, random_state=seed)
        return X @ np.array([[0.6, -0.6], [-0.4, 0.8]]), y
    if name == "multiscale":
        # tight and wide clusters of uneven size: the setting the adaptive cut targets
        centers = np.array([[0, 0], [1.2, 0], [0.6, 1.0], [8, 0], [4, 7]])
        std = np.array([0.25, 0.25, 0.25, 1.6, 1.6])
        y = np.repeat(np.arange(5), [150, 150, 150, 275, 275])
        return centers[y] + std[y, None] * rng.standard_normal((len(y), 2)), y
    if name.startswith("gmm"):
        # random Gaussian mixture: k clusters, uneven sizes and spreads
        d = int(name[3:])
        k = int(rng.integers(4, 11))
        sizes = rng.multinomial(n - 20 * k, rng.dirichlet(np.ones(k))) + 20
        std = np.exp(rng.uniform(np.log(0.2), np.log(2.0), k))
        centers = rng.uniform(-10, 10, (k, d))
        y = np.repeat(np.arange(k), sizes)
        return centers[y] + std[y, None] * rng.standard_normal((len(y), d)), y
    raise ValueError(name)


def classic(name):
    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    load = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "digits": datasets.load_digits,
    }[name]
    X, y = load(return_X_y=True)
    if name == "digits":
        return PCA(10, random_state=0).fit_transform(X), y
    return StandardScaler().fit_transform(X), y


def baselines(X, y, seed=0):
    """Common sklearn clusterings: GMM with BIC-selected k, HDBSCAN, k-means at the true k."""
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.mixture import GaussianMixture

    k_true = len(np.unique(y))
    out = {}
    for cov in ("diag", "full"):
        fits = [
            GaussianMixture(k, covariance_type=cov, random_state=seed, n_init=1).fit(X)
            for k in range(1, 16)
        ]
        best = min(fits, key=lambda g: g.bic(X))
        out[f"gmm_{cov}_ari"] = ari(y, best.predict(X))
        out[f"gmm_{cov}_k"] = best.n_components
    out["hdbscan_sklearn_ari"] = ari(y, HDBSCAN().fit_predict(X))
    out["kmeans_true_k_ari"] = ari(y, KMeans(k_true, n_init=10, random_state=seed).fit_predict(X))
    return out


def dendrogram(X, method, min_samples=5):
    if method != "hdbscan":
        return linkage(X, method)
    # HDBSCAN's tree: single linkage on the mutual-reachability distance
    D = squareform(pdist(X))
    core = np.sort(D, axis=1)[:, min_samples]
    mr = np.maximum(D, np.maximum(core[:, None], core[None, :]))
    np.fill_diagonal(mr, 0)
    return linkage(squareform(mr, checks=False), "single")


SETTINGS = {"default": dict(scale=1.0, n_steps=None), "hot": dict(scale=100.0, n_steps=100_000)}


def sampler(Z, scores, setting, seed):
    """Best cut found by the adaptive cut sampler with one of the SETTINGS."""
    n = len(Z) + 1
    gains = np.abs(scores[n:] - scores[Z[:, 0].astype(int)] - scores[Z[:, 1].astype(int)])
    T = SETTINGS[setting]["scale"] * 0.1 * np.median(gains[gains > 0])
    monotone = bool(np.all(np.diff(Z[:, 2]) >= 0))
    init = "best_level" if monotone else "leaves"
    return ac.adaptive_cut(Z, scores, T, n_steps=SETTINGS[setting]["n_steps"], init=init, rng=seed)


def evaluate(task):
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.metrics import adjusted_rand_score as ari

    dataset, seed = task
    X, y = synthetic(dataset, seed) if seed is not None else classic(dataset)
    k_true = len(np.unique(y))
    base = baselines(X, y, 0 if seed is None else seed)
    rows = []
    for method in METHODS:
        t0 = time.perf_counter()
        Z = dendrogram(X, method)
        t_tree = time.perf_counter() - t0
        scores = ac.gaussian_scores(Z, X)
        try:
            level = ac.best_level_cut(Z, scores)
            row = dict(
                level_k=level.n_clusters,
                level_ari=ari(y, level.labels),
                level_ami=ami(y, level.labels),
                level_value=level.value,
            )
        except ValueError:  # centroid and median trees are not monotone
            row = dict(level_k=np.nan, level_ari=np.nan, level_ami=np.nan, level_value=np.nan)
        for setting in SETTINGS:
            t0 = time.perf_counter()
            res = sampler(Z, scores, setting, 0 if seed is None else seed)
            row.update(
                {
                    f"{setting}_k": res.n_clusters,
                    f"{setting}_ari": ari(y, res.labels),
                    f"{setting}_ami": ami(y, res.labels),
                    f"{setting}_value": res.value,
                    f"{setting}_T": res.T,
                    f"{setting}_time": time.perf_counter() - t0,
                }
            )
        oracle = fcluster(Z, k_true, "maxclust")
        rows.append(
            dict(
                dataset=dataset,
                seed=-1 if seed is None else seed,
                method=method,
                n=len(X),
                d=X.shape[1],
                k_true=k_true,
                B=ac.balancedness(Z) if method not in ("centroid", "median") else np.nan,
                **row,
                oracle_k_ari=ari(y, oracle),
                oracle_k_ami=ami(y, oracle),
                t_tree=t_tree,
                **base,
            )
        )
    return rows


def main():
    import pandas as pd

    synth = ["circles", "moons", "blobs", "varied", "aniso", "multiscale", "gmm2", "gmm5", "gmm10"]
    tasks = [(s, seed) for s in synth for seed in SEEDS]
    tasks += [(c, None) for c in ["iris", "wine", "breast_cancer", "digits"]]
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        rows = [r for part in pool.map(evaluate, tasks) for r in part]
    df = pd.DataFrame(rows)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "clustering.csv", index=False)
    print(f"{len(df)} rows written to {out / 'clustering.csv'}")


if __name__ == "__main__":
    main()
