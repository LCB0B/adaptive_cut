"""Link communities: hierarchical clustering of the edges of a network."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp

from ._linkage import sparse_single_linkage
from ._tree import as_linkage
from .balance import balancedness
from .cut import AdaptiveCutResult, Cut, adaptive_cut, best_level_cut


def _is_networkx(G) -> bool:
    return hasattr(G, "edges") and hasattr(G, "is_directed") and hasattr(G, "is_multigraph")


def _node_label_array(nodes) -> np.ndarray:
    try:
        arr = np.asarray(nodes) if len(nodes) else np.empty(0, dtype=np.int64)
    except ValueError:  # ragged labels such as tuples of different lengths
        arr = None
    # keep a typed array only if numpy did not coerce the labels (e.g. ints to str)
    if arr is not None and arr.ndim == 1 and arr.dtype != object and arr.tolist() == list(nodes):
        return arr
    out = np.empty(len(nodes), dtype=object)
    for t, x in enumerate(nodes):
        out[t] = x
    return out


def _index_labels(labels):
    """Node list (sorted when the labels are comparable) and label -> index map."""
    nodes = list(dict.fromkeys(labels))
    try:
        nodes = sorted(nodes)
    except TypeError:
        pass
    return nodes, {nd: t for t, nd in enumerate(nodes)}


def _edges_from_input(graph, weights, weighted, weight_attr):
    """Return (u, v, w or None, node_labels) with u, v integer node indices."""
    if _is_networkx(graph):
        if graph.is_directed():
            raise ValueError("directed graphs are not supported; pass G.to_undirected()")
        if graph.is_multigraph():
            raise ValueError("multigraphs are not supported; pass nx.Graph(G)")
        if weights is not None:
            raise ValueError("weights= is only used with edge arrays; set edge attributes instead")
        nodes, index = _index_labels(graph)
        if weighted:
            data = [
                (index[x], index[y], w) for x, y, w in graph.edges(data=weight_attr, default=1.0)
            ]
        else:
            data = [(index[x], index[y], 1.0) for x, y in graph.edges()]
        arr = np.array(data, dtype=np.float64).reshape(-1, 3)
        u, v = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
        return u, v, (arr[:, 2] if weighted else None), _node_label_array(nodes)

    if sp.issparse(graph):
        if weights is not None:
            raise ValueError("weights= is only used with edge arrays")
        A = sp.csr_matrix(graph, dtype=np.float64, copy=True)
        if A.shape[0] != A.shape[1]:
            raise ValueError("adjacency matrix must be square")
        A.eliminate_zeros()
        B = A if weighted else (A != 0).astype(np.float64)
        if (B != B.T).nnz:
            raise ValueError("adjacency matrix must be symmetric; use A.maximum(A.T)")
        U = sp.triu(A, k=1).tocoo()
        u, v = U.row.astype(np.int64), U.col.astype(np.int64)
        return u, v, (U.data.astype(np.float64) if weighted else None), np.arange(A.shape[0])

    if isinstance(graph, np.ndarray) and graph.dtype != object:
        if graph.ndim != 2 or graph.shape[1] != 2:
            raise ValueError("edge arrays must have shape (M, 2); pass weights separately")
        labels, inv = np.unique(graph, return_inverse=True)
        inv = inv.reshape(-1, 2).astype(np.int64)
    else:
        # sequences of pairs: index labels in Python so that mixed types
        # (1 and "1") or tuple labels are kept as they are
        pairs = [tuple(e) for e in graph]
        if any(len(e) != 2 for e in pairs):
            raise ValueError("edge lists must contain pairs (u, v); pass weights separately")
        nodes, index = _index_labels([x for e in pairs for x in e])
        inv = np.array([(index[x], index[y]) for x, y in pairs], dtype=np.int64).reshape(-1, 2)
        labels = _node_label_array(nodes)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) != len(inv):
            raise ValueError("weights must have one entry per edge")
    elif weighted:
        raise ValueError("weighted=True needs weights= for edge-array input")
    return inv[:, 0], inv[:, 1], w, labels


def _canonical_edges(u, v, w):
    """Drop self-loops, orient u < v, sort lexicographically, remove duplicates."""
    keep = u != v
    u, v = u[keep], v[keep]
    a, b = np.minimum(u, v), np.maximum(u, v)
    o = np.lexsort((b, a))
    a, b = a[o], b[o]
    first = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])]
    if w is not None:
        w = w[keep][o]
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("edge weights must be finite and positive")
        run = np.cumsum(first) - 1
        if np.any(w != w[first][run]):
            raise ValueError("duplicate edges with different weights")
        w = w[first]
    edges = np.column_stack([a[first], b[first]])
    if len(edges) < 2:
        raise ValueError("need at least 2 distinct edges (self-loops are ignored)")
    return edges, w


def _edge_pairs(edges: np.ndarray, N: int, w):
    """All pairs of edges sharing a node, with their distance 1 - similarity.

    Similarity of e_ik and e_jk is the Tanimoto coefficient of the
    neighbourhood vectors of i and j, where the vector of i has the edge
    weights w_ij and the self term w_ii = mean incident weight of i. With
    unit weights this is the Jaccard index of the inclusive neighbourhoods
    {i} U N(i) and {j} U N(j).
    """
    M = len(edges)
    u, v = edges[:, 0], edges[:, 1]
    own = np.r_[u, v]
    oth = np.r_[v, u]
    eid = np.r_[np.arange(M), np.arange(M)]
    s = np.lexsort((eid, own))
    own, oth, eid = own[s], oth[s], eid[s]
    deg = np.bincount(own, minlength=N)
    ptr = np.r_[0, np.cumsum(deg)]
    cnt = ptr[own + 1] - np.arange(2 * M) - 1  # later incidences at the same node
    first = np.repeat(np.arange(2 * M), cnt)
    second = np.arange(first.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    second += first + 1
    ea, eb = eid[first], eid[second]
    i, j = oth[first], oth[second]
    del first, second

    # Tanimoto is scale invariant: normalising avoids overflow and makes
    # uniform weights reproduce the unweighted (Jaccard) values exactly
    ww = np.ones(M) if w is None else w / w.max()
    idx = (np.r_[u, v].astype(np.int32), np.r_[v, u].astype(np.int32))
    A = sp.csr_matrix((np.r_[ww, ww], idx), shape=(N, N))
    strength = np.asarray(A.sum(axis=1)).ravel()
    self_w = np.divide(strength, deg, out=np.zeros(N), where=deg > 0)
    At = (A + sp.diags(self_w, format="csr")).tocsr()
    A2 = (At @ At).tocsr()
    A2.sort_indices()  # fast element lookup below
    dot = np.asarray(A2[i, j]).ravel()
    del A2
    sq = np.asarray(At.multiply(At).sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = 1.0 - dot / (sq[i] + sq[j] - dot)
    if not np.all(np.isfinite(dist)):
        raise ValueError("edge weight ratios are too extreme for float64 similarities")
    if w is not None:
        # rounding can push the distance of identical neighbourhoods below 0
        dist[dist < 8 * np.finfo(np.float64).eps] = 0.0
        np.minimum(dist, 1.0, out=dist)
    return ea, eb, dist


def partition_density_scores(Z, edges) -> np.ndarray:
    r"""Per-node partition-density scores of a dendrogram over edges.

    For the cluster of node ``v`` with ``m`` edges spanning ``n`` distinct
    nodes, ``scores[v] = 2 m (m - n + 1) / ((n - 1)(n - 2)) / M`` (0 when
    ``n <= 2``), so ``scores[cut.nodes].sum()`` is the partition density
    ``D`` of the edge partition.

    Parameters
    ----------
    Z : array_like, shape (M-1, 4)
        Linkage matrix whose leaf ``i`` is edge ``i``.
    edges : array_like of int, shape (M, 2)
        Endpoints of every edge.
    """
    Z, M = as_linkage(Z)
    edges = np.asarray(edges, dtype=np.int64)
    if edges.shape != (M, 2):
        raise ValueError(f"edges must have shape ({M}, 2), got {edges.shape}")
    m = np.ones(2 * M - 1)
    m[M:] = Z[:, 3]
    nv = np.empty(2 * M - 1)
    nv[:M] = [len({x, y}) for x, y in edges.tolist()]
    sets: list = [{x, y} for x, y in edges.tolist()] + [None] * (M - 1)
    for i, (l, r) in enumerate(Z[:, :2].astype(np.int64).tolist()):
        big, small = sets[l], sets[r]
        if len(big) < len(small):
            big, small = small, big
        big |= small  # small-to-large merge, O(M log M) inserts in total
        sets[M + i] = big
        sets[l] = sets[r] = None
        nv[M + i] = len(big)
    den = (nv - 1) * (nv - 2)
    out = np.zeros(2 * M - 1)
    np.divide(2.0 * m * (m - nv + 1), den * M, out=out, where=nv > 2)
    return out


def _partition_density_temperature(M: int, D: float) -> float:
    """Initial temperature for partition density: clip(0.1/M * f_D, 1e-8, 1e-3)."""
    f_D = 0.3 if D > 0.7 else 0.01 if D < 0.01 else 1.0
    return min(max(0.1 / M * f_D, 1e-8), 1e-3)


@dataclass(frozen=True, eq=False)
class LinkCommunities:
    """Link communities of a network (see :func:`link_communities`).

    Attributes
    ----------
    node_labels : ndarray, shape (N,)
        Original node label of every node index.
    edges : ndarray of int, shape (M, 2)
        Edge endpoints as node indices, ``u < v``, sorted. Edge ``i`` is leaf
        ``i`` of the dendrogram.
    weights : ndarray, shape (M,), or None
    Z : ndarray, shape (M-1, 4)
        Single-linkage dendrogram of the edges (heights are 1 - similarity).
    scores : ndarray, shape (2M-1,)
        Partition-density score of every dendrogram node.
    balancedness : float
        Balancedness of ``Z``.
    level_cut : Cut
        Best single-level cut; ``level_cut.value`` is its partition density.
    adaptive : AdaptiveCutResult or None
        Adaptive cut started from ``level_cut``.
    T : float or None
        Temperature used by the adaptive cut.
    """

    node_labels: np.ndarray
    edges: np.ndarray
    weights: np.ndarray | None
    Z: np.ndarray
    scores: np.ndarray
    balancedness: float
    level_cut: Cut
    adaptive: AdaptiveCutResult | None
    T: float | None

    @property
    def cut(self) -> Cut:
        """The adaptive cut if it was run, else the best single-level cut."""
        return self.adaptive.best if self.adaptive is not None else self.level_cut

    @property
    def partition_density(self) -> float:
        return self.cut.value

    def memberships(self, cut: Cut | None = None):
        """Sparse (N, k) matrix: fraction of each node's edges in each community.

        Rows sum to 1 (0 for isolated nodes).
        """
        cut = self.cut if cut is None else cut
        N = len(self.node_labels)
        deg = np.bincount(self.edges.ravel(), minlength=N)
        rows = np.r_[self.edges[:, 0], self.edges[:, 1]]
        cols = np.r_[cut.labels, cut.labels]
        M = sp.coo_matrix((1.0 / deg[rows], (rows, cols)), shape=(N, cut.n_clusters))
        return sp.csr_array(M.tocsr())

    def communities(self, cut: Cut | None = None) -> list[np.ndarray]:
        """Node labels of every link community (communities overlap)."""
        cut = self.cut if cut is None else cut
        o = np.argsort(cut.labels, kind="stable")
        bounds = np.cumsum(np.bincount(cut.labels, minlength=cut.n_clusters))[:-1]
        return [self.node_labels[np.unique(self.edges[g])] for g in np.split(o, bounds)]

    def __repr__(self) -> str:
        return (
            f"LinkCommunities(N={len(self.node_labels)}, M={len(self.edges)}, "
            f"balancedness={self.balancedness:.3f}, D_level={self.level_cut.value:.4f}, "
            f"D={self.partition_density:.4f}, n_communities={self.cut.n_clusters})"
        )


def link_communities(
    graph,
    *,
    weights=None,
    weighted: bool = False,
    weight_attr: str = "weight",
    adaptive: bool = True,
    T="auto",
    n_steps: int | None = None,
    k_sa: int | None = None,
    rng=None,
    engine: str = "auto",
    trace_every: int | None = None,
) -> LinkCommunities:
    """Link communities of an undirected network, with an adaptive cut.

    Edges are clustered by single linkage on the similarity of edge pairs
    that share a node (Jaccard of the inclusive neighbourhoods of their other
    endpoints, or Tanimoto for weighted networks). The dendrogram is cut at
    the level of highest partition density, then refined by
    :func:`adaptive_cut`.

    Parameters
    ----------
    graph : networkx.Graph, array_like of shape (M, 2), or scipy sparse matrix
        Undirected network: a networkx graph, an edge list of node labels, or
        a symmetric adjacency matrix. Self-loops and duplicate edges are
        ignored.
    weights : array_like, shape (M,), optional
        Edge weights for edge-list input (implies ``weighted=True``).
    weighted : bool
        Use edge weights (networkx attribute ``weight_attr``, matrix values,
        or ``weights``).
    adaptive : bool
        Run the adaptive cut; if False only the single-level cut is computed.
    T : float or "auto"
        Initial temperature. ``"auto"`` is
        ``clip(0.1/M * f_D, 1e-8, 1e-3)`` with ``f_D = 0.3`` if the
        single-level partition density exceeds 0.7, 0.01 if it is below 0.01
        (else 1).
    n_steps, k_sa, rng, engine, trace_every
        Passed to :func:`adaptive_cut`.

    Returns
    -------
    LinkCommunities
    """
    weighted = weighted or weights is not None
    u, v, w, node_labels = _edges_from_input(graph, weights, weighted, weight_attr)
    edges, w = _canonical_edges(u, v, w)
    M = len(edges)
    ea, eb, dist = _edge_pairs(edges, len(node_labels), w)
    Z = sparse_single_linkage(M, ea, eb, dist, fill_height=1.0)
    scores = partition_density_scores(Z, edges)
    B = balancedness(Z)
    level = best_level_cut(Z, scores)
    result = None
    if adaptive:
        if isinstance(T, str) and T == "auto":
            T = _partition_density_temperature(M, level.value)
        result = adaptive_cut(
            Z,
            scores,
            T,
            n_steps=n_steps,
            k_sa=k_sa,
            init=level.nodes,
            rng=rng,
            engine=engine,
            trace_every=trace_every,
        )
        result = replace(result, initial=level)  # same cut, but keeps the level height
        T = result.T
    else:
        T = None
    return LinkCommunities(node_labels, edges, w, Z, scores, B, level, result, T)


__all__ = ["LinkCommunities", "link_communities", "partition_density_scores"]
