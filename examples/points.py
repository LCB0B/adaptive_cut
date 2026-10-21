"""Adaptive cut of a point-cloud dendrogram.

Points are clustered by Ward linkage, and each cluster is scored by how well
one Gaussian describes it (a per-cluster BIC, see ``gaussian_scores``). The
toy data have three tight blobs next to two wide ones: no single height of
the dendrogram separates the tight blobs without also cutting the wide ones
into pieces. Run with ``python examples/points.py``; the classic datasets
need scikit-learn.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage

import adaptive_cut as ac


def gaussian_scores(X, Z, var_floor=0.01):
    """Per-node score: BIC of one diagonal Gaussian fitted to the cluster.

    For a cluster of ``m`` points with per-dimension variances ``v``:
    ``-m/2 * sum(log(2 pi v) + 1) + m log m - (2d + 1)/2 * log N``.
    The sum over the clusters of a cut is the (hard-assignment) BIC of a
    Gaussian mixture with one component per cluster, up to a constant.
    Variances are floored at ``var_floor`` times the variance of the data so
    that singletons get a finite score. The sufficient statistics (count,
    sum, sum of squares) are accumulated bottom-up, so this is O(n d).
    """
    X = np.asarray(X, float)
    N, d = X.shape
    m = np.r_[np.ones(N), Z[:, 3]]
    s1 = np.vstack([X, np.zeros((N - 1, d))])
    s2 = np.vstack([X**2, np.zeros((N - 1, d))])
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        s1[N + i] = s1[a] + s1[b]
        s2[N + i] = s2[a] + s2[b]
    vhat = np.maximum(s2 / m[:, None] - (s1 / m[:, None]) ** 2, 0)
    v = np.maximum(vhat, var_floor * X.var(axis=0))
    loglik = -0.5 * (m[:, None] * (np.log(2 * np.pi * v) + vhat / v)).sum(axis=1)
    return loglik + m * np.log(m) - 0.5 * (2 * d + 1) * np.log(N)


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


def cluster(X, method="ward"):
    """Ward dendrogram, Gaussian scores, best single-level and best adaptive cut."""
    Z = linkage(X, method)
    scores = gaussian_scores(X, Z)
    return Z, ac.best_level_cut(Z, scores), ac.optimal_cut(Z, scores)


def run(seed=0):
    X, y = blobs(seed)
    Z, level, best = cluster(X)
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
