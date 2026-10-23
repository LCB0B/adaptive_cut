"""Adaptive cut of a point-cloud dendrogram.

Points are clustered by Ward linkage, and each cluster is scored by how well
one Gaussian describes it (a per-cluster BIC, ``adaptive_cut.gaussian_scores``). The
toy data have three tight blobs next to two wide ones: no single height of
the dendrogram separates the tight blobs without also cutting the wide ones
into pieces. Run with ``python examples/points.py``; the classic datasets
need scikit-learn.

On the blobs the sampler needs a high temperature: from the single-level cut
(3 clusters) to the 5-cluster cut it has to pass through cuts about 30 BIC
units worse, while ``T="auto"`` is about 2 units, so the default chain stays
at the single-level cut.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage

import adaptive_cut as ac


def blobs(seed=0):
    """Three tight blobs (150 points each) and two wide ones (300 each)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [1.2, 0], [0.6, 1.0], [8, 0], [4, 7]])
    std = np.array([0.25, 0.25, 0.25, 1.6, 1.6])
    y = np.repeat(np.arange(5), [150, 150, 150, 300, 300])
    X = centers[y] + std[y, None] * rng.standard_normal((len(y), 2))
    return X, y


def ari(a, b):
    """Adjusted Rand index between two labelings."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    pairs = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
    n, ra, rb = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = ra * rb / pairs(np.array([len(a)]))
    return (n - expected) / ((ra + rb) / 2 - expected)


def cluster(X, method="ward", T="auto", n_steps=None, seed=0):
    """Dendrogram, Gaussian scores, best single-level cut and adaptive cut."""
    Z = linkage(X, method)
    scores = ac.gaussian_scores(Z, X)
    result = ac.adaptive_cut(Z, scores, T, n_steps=n_steps, rng=seed)
    return Z, ac.best_level_cut(Z, scores), result.best


def run(seed=0):
    X, y = blobs(seed)
    Z, level, best = cluster(X, T=160, n_steps=100_000, seed=seed)
    print(f"B = {ac.balancedness(Z):.2f}")
    print(f"single-level cut: {level.n_clusters} clusters, ARI = {ari(y, level.labels):.2f}")
    print(f"adaptive cut:     {best.n_clusters} clusters, ARI = {ari(y, best.labels):.2f}")
    return dict(X=X, y=y, Z=Z, level=level, adaptive=best)


def classic():
    """Iris and wine (standardised) and digits (PCA to 10 dimensions), Ward linkage."""
    from sklearn.datasets import load_digits, load_iris, load_wine
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rows = []
    for name, load in [("iris", load_iris), ("wine", load_wine), ("digits", load_digits)]:
        X, y = load(return_X_y=True)
        if name == "digits":
            X = PCA(10, random_state=0).fit_transform(X)
        else:
            X = StandardScaler().fit_transform(X)
        Z, level, best = cluster(X)
        rows.append(
            (
                name,
                len(X),
                ac.balancedness(Z),
                level,
                best,
                ari(y, level.labels),
                ari(y, best.labels),
            )
        )
        print(
            f"{name:>7}: n={len(X):4d}  B={rows[-1][2]:.2f}  "
            f"single-level {level.n_clusters:2d} clusters ARI={rows[-1][5]:.2f}  "
            f"adaptive {best.n_clusters:2d} clusters ARI={rows[-1][6]:.2f}"
        )
    return rows


if __name__ == "__main__":
    run()
    try:
        classic()
    except ImportError:
        print("install scikit-learn for the iris, wine and digits examples")
