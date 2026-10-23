import numpy as np
import pytest
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage

import adaptive_cut as ac

METHODS = ["single", "complete", "average", "ward", "centroid"]


def _gaussian_direct(X, covariance, var_floor=0.01):
    """Per-cluster BIC computed from scratch for one cluster at a time."""
    N, d = X.shape
    floor = var_floor * X.var(axis=0)

    def score(idx):
        P = X[idx]
        m = len(P)
        if covariance == "full":
            ch = np.cov(P, rowvar=False, bias=True).reshape(d, d)
            C = ch + np.diag(floor)
            ll = (
                -0.5
                * m
                * (
                    d * np.log(2 * np.pi)
                    + np.linalg.slogdet(C)[1]
                    + np.trace(np.linalg.solve(C, ch))
                )
            )
            p = d + d * (d + 1) / 2 + 1
        elif covariance == "spherical":
            vh = P.var(axis=0).mean()
            v = max(vh, floor.mean())
            ll = -0.5 * m * d * (np.log(2 * np.pi * v) + vh / v)
            p = d + 2
        else:
            vh = P.var(axis=0)
            v = np.maximum(vh, floor)
            ll = -0.5 * m * (np.log(2 * np.pi * v) + vh / v).sum()
            p = 2 * d + 1
        return ll + m * np.log(m) - 0.5 * p * np.log(N)

    return score


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("covariance", ["diag", "spherical", "full"])
def test_gaussian_scores_match_direct_computation(method, covariance):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((70, 3)) * [1.0, 3.0, 0.2] + 50.0  # far from the origin
    Z = linkage(X, method)
    expected = ac.node_scores(Z, _gaussian_direct(X, covariance))
    np.testing.assert_allclose(
        ac.gaussian_scores(Z, X, covariance=covariance), expected, rtol=1e-9, atol=1e-8
    )


def test_gaussian_scores_recover_separated_clusters():
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    y = np.repeat([0, 1, 2], 60)
    X = centers[y] + rng.standard_normal((180, 2))
    Z = linkage(X, "average")
    cut = ac.adaptive_cut(Z, ac.gaussian_scores(Z, X), "auto", rng=0).best
    assert cut.n_clusters == 3
    assert len(set(zip(cut.labels, y))) == 3


def test_gaussian_scores_accept_constant_columns_and_1d():
    X = np.c_[np.arange(10.0), np.zeros(10)]
    Z = linkage(X, "single")
    assert np.all(np.isfinite(ac.gaussian_scores(Z, X)))
    assert ac.gaussian_scores(Z, X[:, 0]).shape == (19,)


@pytest.mark.parametrize(
    "kwargs, match",
    [({"covariance": "tied"}, "covariance"), ({"var_floor": 0.0}, "var_floor")],
)
def test_gaussian_scores_reject_bad_arguments(kwargs, match):
    X = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError, match=match):
        ac.gaussian_scores(linkage(X, "ward"), X, **kwargs)


def test_gaussian_scores_reject_mismatched_X():
    X = np.random.default_rng(0).standard_normal((10, 2))
    Z = linkage(X, "ward")
    with pytest.raises(ValueError, match="one per leaf"):
        ac.gaussian_scores(Z, X[:9])
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ac.gaussian_scores(Z, X)


def _random_graph(n, seed, weighted=True, loops=True):
    rng = np.random.default_rng(seed)
    W = sp.random(n, n, density=0.1, random_state=seed)
    if weighted:
        W.data = rng.uniform(0.5, 2.0, W.nnz)
    else:
        W.data[:] = 1.0
    A = (W + W.T).tolil()
    if loops:
        for i in range(0, n, 7):
            A[i, i] = rng.uniform(1, 3)
    return A.tocsr()


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("resolution", [1.0, 0.5])
def test_modularity_scores_match_direct_computation(seed, resolution):
    n = 50
    A = _random_graph(n, seed)
    D = A.toarray()
    deg = D.sum(axis=1)
    two_m = deg.sum()
    Z = linkage(np.random.default_rng(seed).standard_normal((n, 2)), METHODS[seed])

    def q(idx):
        return D[np.ix_(idx, idx)].sum() / two_m - resolution * (deg[idx].sum() / two_m) ** 2

    expected = ac.node_scores(Z, q)
    np.testing.assert_allclose(
        ac.modularity_scores(Z, A, resolution=resolution), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        ac.modularity_scores(Z, D, resolution=resolution), expected, atol=1e-12
    )


def test_modularity_of_a_cut_equals_networkx():
    nx = pytest.importorskip("networkx")
    G = nx.karate_club_graph()
    A = nx.to_scipy_sparse_array(G, nodelist=range(34), weight=None)
    Z = linkage(np.random.default_rng(0).standard_normal((34, 3)), "ward")
    cut = ac.adaptive_cut(Z, ac.modularity_scores(Z, A), "auto", rng=0).best
    communities = [set(np.flatnonzero(cut.labels == c)) for c in range(cut.n_clusters)]
    assert cut.value == pytest.approx(nx.community.modularity(G, communities, weight=None))


def test_modularity_scores_reject_bad_graphs():
    Z = linkage(np.random.default_rng(0).standard_normal((4, 2)), "ward")
    A = np.ones((4, 4))
    with pytest.raises(ValueError, match="symmetric"):
        ac.modularity_scores(Z, np.triu(A))
    with pytest.raises(ValueError, match="leaves"):
        ac.modularity_scores(Z, np.ones((5, 5)))
    with pytest.raises(ValueError, match="no edges"):
        ac.modularity_scores(Z, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="non-negative"):
        ac.modularity_scores(Z, -A)
