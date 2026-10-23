"""Ready-made per-node objectives for point clouds and networks.

Each function returns ``scores`` of shape ``(2n-1,)`` for a scipy linkage
matrix, so that ``scores[cut].sum()`` is the objective of a cut. Pass them
to :func:`adaptive_cut` or :func:`best_level_cut`.
All of them run in O(n d) (``covariance="full"``: O(n d^2)) or O(n log n + m)
time, using that the leaves of every dendrogram node are a contiguous block
of the dendrogram's leaf order.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._tree import as_linkage, build_tree


def _block_sums(tree, values: np.ndarray) -> np.ndarray:
    """Sum of ``values[leaf]`` over the leaves of every node, shape (2n-1, ...)."""
    csum = np.concatenate(
        [np.zeros((1,) + values.shape[1:]), np.cumsum(values[tree.order], axis=0)]
    )
    return csum[tree.start + tree.size] - csum[tree.start]


def gaussian_scores(Z, X, *, covariance: str = "diag", var_floor: float = 0.01) -> np.ndarray:
    """Per-node BIC of one Gaussian fitted to the points of the cluster.

    For a cut, ``scores[cut].sum()`` is (up to a constant) the BIC of a
    Gaussian mixture with one component per cluster and hard assignments, so
    maximising it balances fit against the number of clusters. A cluster of
    ``m`` of the ``N`` points scores

    ``loglik(cluster) + m log m - (p / 2) log N``,

    with ``p`` the number of parameters of one component (mean, covariance
    and weight) and the mean and covariance set to their maximum-likelihood
    values.

    Parameters
    ----------
    Z : array_like, shape (N-1, 4)
        scipy linkage matrix of the rows of ``X``.
    X : array_like, shape (N, d)
        The clustered points, in the order used to build ``Z``.
    covariance : {"diag", "spherical", "full"}
        Covariance model of each cluster: one variance per dimension, one
        variance for all dimensions, or a full covariance matrix.
    var_floor : float
        Variances are floored at ``var_floor`` times the variance of the data
        (per dimension), so that singletons and flat clusters have a finite
        score. For ``"full"`` the floor is added to the diagonal.

    Returns
    -------
    ndarray, shape (2N-1,)
    """
    Z, n = as_linkage(Z)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] != n:
        raise ValueError(f"X must have shape (n, d) with n = {n} rows, one per leaf of Z")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if covariance not in ("diag", "spherical", "full"):
        raise ValueError('covariance must be "diag", "spherical" or "full"')
    if not var_floor > 0:
        raise ValueError("var_floor must be > 0")
    tree = build_tree(Z, n)
    d = X.shape[1]
    m = tree.size.astype(np.float64)
    data_var = X.var(axis=0)
    data_var = np.where(data_var > 0, data_var, 1.0)
    floor = var_floor * data_var
    # center first: the sums of squares then do not lose precision far from the origin
    Xc = X - X.mean(axis=0)
    s1 = _block_sums(tree, Xc)
    mean = s1 / m[:, None]
    if covariance == "full":
        s2 = _block_sums(tree, Xc[:, :, None] * Xc[:, None, :])
        covhat = s2 / m[:, None, None] - mean[:, :, None] * mean[:, None, :]
        cov = covhat + np.diag(floor)[None]
        _, logdet = np.linalg.slogdet(cov)
        trace = np.einsum("kij,kji->k", np.linalg.inv(cov), covhat)
        loglik = -0.5 * m * (d * np.log(2 * np.pi) + logdet + trace)
        n_params = d + d * (d + 1) / 2 + 1
    else:
        vhat = np.maximum(_block_sums(tree, Xc**2) / m[:, None] - mean**2, 0.0)
        if covariance == "spherical":
            vhat = vhat.mean(axis=1, keepdims=True)
            floor = floor.mean(keepdims=True)
            n_params = d + 2
        else:
            n_params = 2 * d + 1
        v = np.maximum(vhat, floor)
        k = vhat.shape[1]
        loglik = -0.5 * m * ((np.log(2 * np.pi * v) + vhat / v).sum(axis=1) * (d / k))
    return loglik + m * np.log(m) - 0.5 * n_params * np.log(n)


def _adjacency(A) -> sp.csr_array:
    if hasattr(A, "edges") and hasattr(A, "is_directed"):
        raise TypeError(
            "pass an adjacency matrix, e.g. nx.to_scipy_sparse_array(G, nodelist=nodes), "
            "with rows in the order of the leaves of Z"
        )
    A = sp.csr_array(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix")
    if (abs(A - A.T) > 1e-12 * max(1.0, abs(A).max())).nnz:
        raise ValueError("A must be symmetric (undirected graph); try A.maximum(A.T)")
    if A.nnz and A.data.min() < 0:
        raise ValueError("edge weights must be non-negative")
    return A


def modularity_scores(Z, A, *, resolution: float = 1.0) -> np.ndarray:
    """Per-node modularity of a node dendrogram of a network.

    For a cut, ``scores[cut].sum()`` is the modularity
    ``Q = sum_C [ w(C) / 2m - resolution * (d(C) / 2m)^2 ]`` of the partition,
    with ``w(C)`` the total weight of the edges inside community ``C``
    (counted in both directions, self-loops once), ``d(C)`` its total degree
    and ``2m`` the total degree of the graph.

    Parameters
    ----------
    Z : array_like, shape (n-1, 4)
        scipy linkage matrix whose leaves are the nodes of the graph, e.g.
        ``sknetwork.hierarchy.Paris().fit_predict(A)`` or
        ``scipy.cluster.hierarchy.linkage`` of a node embedding.
    A : array_like or sparse matrix, shape (n, n)
        Symmetric non-negative adjacency matrix, row ``i`` for leaf ``i``.
    resolution : float
        Resolution parameter gamma (1 is the standard modularity).

    Returns
    -------
    ndarray, shape (2n-1,)
    """
    Z, n = as_linkage(Z)
    A = _adjacency(A)
    if A.shape[0] != n:
        raise ValueError(f"A has {A.shape[0]} nodes but Z has {n} leaves")
    deg = np.asarray(A.sum(axis=1)).ravel()
    two_m = deg.sum()
    if two_m <= 0:
        raise ValueError("the graph has no edges")
    tree = build_tree(Z, n)
    pos = np.empty(n, dtype=np.int64)
    pos[tree.order] = np.arange(n)
    # boundary[p]: merge row joining the leaves at order positions p and p+1
    rows = np.arange(n - 1)
    boundary = np.empty(max(n - 1, 1), dtype=np.int64)
    boundary[tree.start[n + rows] + tree.size[tree.left] - 1] = rows
    # every edge u < w first falls inside a cluster at the latest merge between its
    # endpoints in leaf order: a range-maximum query on ``boundary``
    U = sp.triu(A, k=1).tocoo()
    p, q = np.sort(np.stack([pos[U.row], pos[U.col]]), axis=0)
    lca = _range_max(boundary, p, q - 1) if U.nnz else np.empty(0, dtype=np.int64)
    at = np.zeros(max(n - 1, 1))
    np.add.at(at, tree.start[n + lca] + tree.size[tree.left[lca]] - 1, 2 * U.data)
    inside = _block_sums(tree, A.diagonal())
    csum = np.r_[0.0, np.cumsum(at)]
    internal = slice(n, 2 * n - 1)
    inside[internal] += (
        csum[tree.start[internal] + tree.size[internal] - 1] - csum[tree.start[internal]]
    )
    return inside / two_m - resolution * (_block_sums(tree, deg) / two_m) ** 2


def _range_max(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``max(values[lo[i]:hi[i]+1])`` for every i (sparse table, lo <= hi)."""
    table = [values]
    span = 1
    while 2 * span <= len(values):
        prev = table[-1]
        table.append(np.maximum(prev[:-span], prev[span:]))
        span *= 2
    length = hi - lo + 1
    k = np.floor(np.log2(length)).astype(np.int64)
    out = np.empty(len(lo), dtype=values.dtype)
    for j in np.unique(k):
        sel = k == j
        t = table[j]
        out[sel] = np.maximum(t[lo[sel]], t[hi[sel] - (1 << j) + 1])
    return out
