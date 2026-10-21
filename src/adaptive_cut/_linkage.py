"""Single linkage on a sparse distance graph."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree

from . import _kernels


def sparse_single_linkage(n: int, i, j, d, *, fill_height: float | None = None) -> np.ndarray:
    """Single-linkage dendrogram of ``n`` items from sparse pairwise distances.

    Pairs ``(i[t], j[t])`` at distance ``d[t]`` are merged in Kruskal order,
    ties broken by ``(d, min(i, j), max(i, j))``, so the result is
    deterministic. Disconnected components are joined at ``fill_height``
    (default ``max(d) + 1``) in order of their smallest item. Returns a valid,
    monotone scipy linkage matrix of shape ``(n - 1, 4)``.
    """
    if n < 2:
        raise ValueError("need at least 2 items")
    i = np.asarray(i, dtype=np.int64).ravel()
    j = np.asarray(j, dtype=np.int64).ravel()
    d = np.asarray(d, dtype=np.float64).ravel()
    if not (len(i) == len(j) == len(d)):
        raise ValueError("i, j and d must have the same length")
    if np.any(~np.isfinite(d)) or np.any(d < 0):
        raise ValueError("distances must be finite and non-negative")
    if len(i) and (min(i.min(), j.min()) < 0 or max(i.max(), j.max()) >= n):
        raise ValueError(f"item ids must lie in [0, {n})")
    if n >= 2**31:
        raise ValueError("at most 2**31 - 1 items are supported")
    a, b = np.minimum(i, j), np.maximum(i, j)
    keep = a != b
    a, b, d = a[keep], b[keep], d[keep]
    if fill_height is None:
        fill_height = float(d.max()) + 1.0 if len(d) else 1.0
    elif len(d) and fill_height < d.max():
        raise ValueError("fill_height must be >= the largest distance")

    Z = np.zeros((n - 1, 4))
    uf = np.arange(n, dtype=np.int64)
    ufsize = np.ones(n, dtype=np.int64)
    cid = np.arange(n, dtype=np.int64)
    row = 0
    if len(d):
        # Kruskal order; the minimum spanning tree over unique positive rank
        # weights selects exactly the merging pairs (and keeps d == 0 pairs,
        # which csgraph would treat as missing edges)
        o = np.lexsort((b, a, d))
        a, b, d = a[o], b[o], d[o]
        _, firsts = np.unique(a * n + b, return_index=True)
        if len(firsts) < len(a):  # duplicate pairs: keep the first, i.e. smallest d
            sel = np.sort(firsts)
            a, b, d = a[sel], b[sel], d[sel]
        rank = np.arange(1, len(d) + 1, dtype=np.float64)
        G = sp.csr_matrix((rank, (a.astype(np.int32), b.astype(np.int32))), shape=(n, n))
        T = minimum_spanning_tree(G).tocoo()
        t = np.argsort(T.data)
        r = T.data[t].astype(np.int64) - 1
        row = _kernels.call("union_find", "python", n, a[r], b[r], d[r], uf, ufsize, cid, Z, row)
    if row < n - 1:
        # chain the components in order of their smallest item
        roots = uf.copy()
        while True:
            nxt = roots[roots]
            if np.array_equal(nxt, roots):
                break
            roots = nxt
        _, reps = np.unique(roots, return_index=True)
        reps = np.sort(reps)
        ra = np.full(len(reps) - 1, reps[0], dtype=np.int64)
        h = np.full(len(reps) - 1, float(fill_height))
        row = _kernels.call("union_find", "python", n, ra, reps[1:], h, uf, ufsize, cid, Z, row)
    return Z
