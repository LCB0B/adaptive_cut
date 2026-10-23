"""Single-level and adaptive (multi-level) cuts of a dendrogram."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _kernels
from ._tree import Tree, as_linkage, build_tree, cut_from_nodes, is_monotone, level_ends


@dataclass(frozen=True, eq=False)
class Cut:
    """A partition of the leaves into whole dendrogram subtrees.

    Attributes
    ----------
    nodes : ndarray of int, shape (k,)
        Dendrogram node ids (scipy numbering) of the clusters; ``nodes[j]`` is
        cluster ``j``. Clusters are ordered by their smallest leaf.
    labels : ndarray of int, shape (n,)
        Cluster index of every leaf. Labels are canonical (cluster of leaf 0
        is 0, new clusters numbered in order of first appearance), so two
        cuts are the same partition iff ``np.array_equal(a.labels, b.labels)``.
    value : float
        Objective value, the sum of the node scores of the cut.
    height : float or None
        For single-level cuts, the merge height of the level:
        ``fcluster(Z, height, "distance")`` gives the same partition.
        ``-inf`` is the level of all singletons. ``None`` for other cuts.
    """

    nodes: np.ndarray
    labels: np.ndarray
    value: float
    height: float | None = None

    @property
    def n_clusters(self) -> int:
        return len(self.nodes)

    @property
    def sizes(self) -> np.ndarray:
        """Number of leaves in each cluster."""
        return np.bincount(self.labels, minlength=len(self.nodes))

    def __repr__(self) -> str:
        h = "" if self.height is None else f", height={self.height:.6g}"
        return f"Cut(n_clusters={self.n_clusters}, value={self.value:.6g}{h})"


@dataclass(frozen=True, eq=False)
class AdaptiveCutResult:
    """Outcome of :func:`adaptive_cut`.

    ``best`` is the best cut visited by the chain and is the answer;
    ``nodes``, ``labels``, ``value`` and ``n_clusters`` forward to it.
    ``trace[i]`` is the objective of the current state after step
    ``(i + 1) * trace_every - 1``.
    """

    best: Cut
    final: Cut
    initial: Cut
    trace: np.ndarray
    trace_every: int
    n_accepted: int
    T: float
    n_steps: int
    k_sa: int
    engine: str

    @property
    def nodes(self) -> np.ndarray:
        return self.best.nodes

    @property
    def labels(self) -> np.ndarray:
        return self.best.labels

    @property
    def value(self) -> float:
        return self.best.value

    @property
    def n_clusters(self) -> int:
        return self.best.n_clusters

    @property
    def gain(self) -> float:
        """Improvement of the best cut over the initial cut."""
        return self.best.value - self.initial.value

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / max(self.n_steps, 1)

    def __repr__(self) -> str:
        return (
            f"AdaptiveCutResult(value={self.value:.6g}, n_clusters={self.n_clusters}, "
            f"initial={self.initial.value:.6g}, n_steps={self.n_steps}, T={self.T:.3g}, "
            f"engine={self.engine!r})"
        )


def _check_scores(scores, n: int) -> np.ndarray:
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    if scores.shape != (2 * n - 1,):
        raise ValueError(f"scores must have shape (2n-1,) = ({2 * n - 1},), got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return scores


def _make_cut(tree: Tree, nodes, scores, height=None) -> Cut:
    c = cut_from_nodes(tree, nodes, scores)
    return Cut(c.nodes, c.labels, c.value, height)


def node_scores(Z, func) -> np.ndarray:
    """Evaluate ``func`` on the leaf set of every dendrogram node.

    Parameters
    ----------
    Z : array_like, shape (n-1, 4)
        scipy linkage matrix.
    func : callable
        ``func(leaves) -> float`` where ``leaves`` is a read-only int array of
        the leaf ids under the node (in dendrogram order, not sorted).

    Returns
    -------
    ndarray, shape (2n-1,)
        ``scores[v] = func(leaves of v)`` for every node id ``v``. The
        objective of a cut is then ``scores[cut.nodes].sum()``, so ``func``
        must be a per-cluster score whose sum over clusters is the objective.

    Notes
    -----
    ``func`` is called ``2n - 1`` times on sets whose total size is the sum
    of all subtree sizes, which is O(n^2) on chain-like dendrograms. For
    large trees compute the scores bottom-up instead.
    """
    Z, n = as_linkage(Z)
    tree = build_tree(Z, n)
    order = tree.order.copy()
    order.flags.writeable = False
    out = np.empty(2 * n - 1)
    for v in range(2 * n - 1):
        out[v] = func(order[tree.start[v] : tree.start[v] + tree.size[v]])
    return out


def _level_nodes(Z: np.ndarray, n: int, scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Nodes and height of the best single-level cut (first maximum wins)."""
    left = Z[:, 0].astype(np.int64)
    right = Z[:, 1].astype(np.int64)
    ends = level_ends(Z)
    f0 = math.fsum(scores[:n])
    vals = np.r_[f0, f0 + np.cumsum(scores[n:] - scores[left] - scores[right])[ends]]
    b = int(np.argmax(vals))
    if b == 0:
        return np.arange(n), -math.inf
    i = ends[b - 1]
    rows = np.arange(n - 1)
    created = np.r_[np.full(n, -1), rows]
    parent_row = np.full(2 * n - 1, n - 1)
    parent_row[left] = rows
    parent_row[right] = rows
    return np.flatnonzero((created <= i) & (parent_row > i)), float(Z[i, 2])


def best_level_cut(Z, scores) -> Cut:
    """Best single-level ("horizontal") cut of a dendrogram.

    Every distinct merge height of ``Z`` (plus the all-singletons level) is a
    candidate; the level with the highest ``sum(scores[cut])`` is returned.
    Tied merges are applied together, as in ``fcluster(Z, h, "distance")``.

    Parameters
    ----------
    Z : array_like, shape (n-1, 4)
        scipy linkage matrix with non-decreasing merge heights.
    scores : array_like, shape (2n-1,)
        Additive per-node objective (see :func:`node_scores`).
    """
    Z, n = as_linkage(Z, monotone=True)
    scores = _check_scores(scores, n)
    nodes, height = _level_nodes(Z, n, scores)
    return _make_cut(build_tree(Z, n), nodes, scores, height)


def _auto_temperature(Z: np.ndarray, n: int, scores: np.ndarray) -> float:
    left = Z[:, 0].astype(np.int64)
    right = Z[:, 1].astype(np.int64)
    gains = np.abs(scores[n:] - scores[left] - scores[right])
    gains = gains[gains > 1e-12 * np.max(np.abs(scores))]
    if gains.size == 0:
        raise ValueError("the objective is constant on all cuts; T='auto' is undefined")
    return 0.1 * float(np.median(gains))


def adaptive_cut(
    Z,
    scores,
    T,
    *,
    n_steps: int | None = None,
    k_sa: int | None = None,
    init="best_level",
    rng=None,
    engine: str = "auto",
    trace_every: int | None = None,
    _chunk: int | None = None,
) -> AdaptiveCutResult:
    """Search for the best multi-level cut of a dendrogram.

    Maximises ``f(cut) = sum(scores[cut])`` over all cuts of ``Z`` (sets of
    subtrees that partition the leaves, at any mix of heights) with a
    reversible-jump Metropolis-Hastings chain and simulated annealing. Each
    step picks a current cluster uniformly and either merges it with its
    sibling into the parent (collapsing the sibling if it is split) or splits
    it into a random sub-cut drawn with probability ``2^-|s| / Z_c``. The
    temperature is ``T`` up to step ``k_sa`` and ``T / ln(k - k_sa + 2)``
    after it. The best cut visited is returned.

    Parameters
    ----------
    Z : array_like, shape (n-1, 4)
        scipy linkage matrix (monotone heights required only for
        ``init="best_level"``).
    scores : array_like, shape (2n-1,)
        Per-node score, ``scores[v]`` being the contribution of the cluster
        formed by the leaves under node ``v``. To minimise, negate it.
    T : float or "auto"
        Initial temperature, in units of ``scores``. ``"auto"`` uses
        ``0.1 * median |scores[v] - scores[left] - scores[right]|`` over the
        internal nodes where this merge gain is non-zero.
    n_steps : int, optional
        Number of proposals; default ``max(10_000, 10 n)``.
    k_sa : int, optional
        Step at which annealing starts; default ``2 * n_steps // 3``. Use
        ``k_sa >= n_steps`` for a constant temperature.
    init : {"best_level", "leaves", "root"} or array of node ids
        Starting cut. ``"best_level"`` is the best single-level cut.
    rng : None, int, numpy.random.Generator or SeedSequence
        Source of randomness, passed to ``numpy.random.default_rng``.
    engine : {"auto", "numba", "python"}
        ``"auto"`` uses numba when it imports cleanly. Both engines give
        identical results for the same ``rng``.
    trace_every : int, optional
        Record the objective every ``trace_every`` steps; default
        ``max(1, n_steps // 100_000)``.

    Returns
    -------
    AdaptiveCutResult
    """
    Z, n = as_linkage(Z)
    scores = _check_scores(scores, n)
    if isinstance(T, str):
        if T != "auto":
            raise ValueError(f"T must be a positive float or 'auto', got {T!r}")
        T = _auto_temperature(Z, n, scores)
    T = float(T)
    if not (math.isfinite(T) and T > 0):
        raise ValueError(f"T must be a finite positive number, got {T}")
    n_steps = max(10_000, 10 * n) if n_steps is None else int(n_steps)
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    k_sa = 2 * n_steps // 3 if k_sa is None else int(k_sa)
    trace_every = max(1, n_steps // 100_000) if trace_every is None else int(trace_every)
    if trace_every < 1:
        raise ValueError("trace_every must be >= 1")
    engine = _kernels.resolve_engine(engine)
    tree = build_tree(Z, n)

    if isinstance(init, str):
        if init == "best_level":
            if not is_monotone(Z):
                raise ValueError(
                    "init='best_level' needs non-decreasing merge heights; "
                    "use init='leaves' or 'root'"
                )
            init_nodes, height = _level_nodes(Z, n, scores)
        elif init == "leaves":
            init_nodes, height = np.arange(n), None
        elif init == "root":
            init_nodes, height = np.array([2 * n - 2]), None
        else:
            raise ValueError(f"unknown init {init!r}")
    else:
        init_nodes, height = init, None
    initial = _make_cut(tree, init_nodes, scores, height)

    k = initial.n_clusters
    cut = np.zeros(n, dtype=np.int64)
    cut[:k] = initial.nodes
    pos = np.full(2 * n - 1, -1, dtype=np.int64)
    pos[initial.nodes] = np.arange(k)
    best = np.zeros(n, dtype=np.int64)
    trace = np.empty(n_steps // trace_every)
    stack = np.zeros(n + 1, dtype=np.int64)
    rem = np.zeros(n + 1, dtype=np.int64)
    add = np.zeros(n + 1, dtype=np.int64)
    ist = np.array([0, k, 0, 1, k, 0], dtype=np.int64)
    fst = np.array([initial.value, initial.value])

    rng = np.random.default_rng(rng)
    chunk = max(1 << 16, 4 * (n + 4)) if _chunk is None else int(_chunk)
    u = rng.random(chunk)
    while True:
        _kernels.call(
            "chain",
            engine,
            tree.left,
            tree.right,
            tree.parent,
            n,
            scores,
            tree.logZc,
            tree.p_stop,
            tree.p_both,
            tree.p_lcum,
            cut,
            pos,
            best,
            trace,
            stack,
            rem,
            add,
            u,
            ist,
            fst,
            T,
            k_sa,
            n_steps,
            trace_every,
        )
        if ist[0] >= n_steps:
            break
        # carry the unused uniforms over, so results do not depend on the chunk size
        u = np.concatenate([u[ist[2] :], rng.random(chunk)])
        ist[2] = 0

    k = int(ist[1])
    final = _make_cut(tree, cut[:k], scores)
    snapshot = final if ist[3] == 1 else _make_cut(tree, best[: ist[4]], scores)
    # the chain tracks a running sum; settle the best visited cut on exact sums
    # (on ties keep the initial cut, which carries the level height)
    best_cut = max((snapshot, final), key=lambda c: c.value)
    if best_cut.value <= initial.value:
        best_cut = initial
    return AdaptiveCutResult(
        best=best_cut,
        final=final,
        initial=initial,
        trace=trace,
        trace_every=trace_every,
        n_accepted=int(ist[5]),
        T=T,
        n_steps=n_steps,
        k_sa=k_sa,
        engine=engine,
    )
