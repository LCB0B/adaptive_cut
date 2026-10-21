"""Linkage-matrix validation and flat-array tree tables."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from . import _kernels


def as_linkage(Z, *, monotone: bool = False) -> tuple[np.ndarray, int]:
    """Validate a scipy linkage matrix and return ``(Z as float64, n_leaves)``.

    Checks run in O(n) and the error message names the failing check.
    """
    Z = np.array(Z, dtype=np.float64, order="C", copy=True)
    if Z.ndim != 2 or Z.shape[1] != 4 or Z.shape[0] < 1:
        raise ValueError(f"linkage matrix must have shape (n-1, 4) with n >= 2, got {Z.shape}")
    if not np.all(np.isfinite(Z)):
        raise ValueError("linkage matrix contains non-finite values")
    n = Z.shape[0] + 1
    ch = Z[:, :2]
    if np.any(ch != np.floor(ch)):
        raise ValueError("linkage child ids must be integers")
    ch = ch.astype(np.int64)
    if np.any(ch < 0) or np.any(ch >= (n + np.arange(n - 1))[:, None]):
        raise ValueError("linkage row i may only merge leaves or clusters created by rows < i")
    if np.any(np.bincount(ch.ravel(), minlength=2 * n - 1)[: 2 * n - 2] != 1):
        raise ValueError("every leaf and non-root cluster must be merged exactly once")
    size = np.ones(2 * n - 1)
    size[n:] = Z[:, 3]
    if np.any(Z[:, 3] != size[ch[:, 0]] + size[ch[:, 1]]):
        raise ValueError("linkage column 3 must hold the size of the merged cluster")
    if np.any(Z[:, 2] < 0):
        raise ValueError("merge heights must be non-negative")
    if monotone and np.any(np.diff(Z[:, 2]) < 0):
        raise ValueError("merge heights must be non-decreasing")
    return Z, n


def level_ends(Z: np.ndarray) -> np.ndarray:
    """Index of the last row of each distinct merge height (rows sorted by height)."""
    return np.r_[np.flatnonzero(np.diff(Z[:, 2]) != 0), len(Z) - 1]


def is_monotone(Z: np.ndarray) -> bool:
    return bool(np.all(np.diff(Z[:, 2]) >= 0))


class Tree(NamedTuple):
    n: int
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray
    size: np.ndarray
    start: np.ndarray
    order: np.ndarray
    minleaf: np.ndarray
    logZ: np.ndarray
    logZc: np.ndarray
    p_stop: np.ndarray
    p_both: np.ndarray
    p_lcum: np.ndarray


def build_tree(Z: np.ndarray, n: int) -> Tree:
    """Per-node tables of a validated linkage (always built by the Python kernel)."""
    m = 2 * n - 1
    left = np.ascontiguousarray(Z[:, 0], dtype=np.int64)
    right = np.ascontiguousarray(Z[:, 1], dtype=np.int64)
    ints = [np.zeros(m, dtype=np.int64) for _ in range(5)]
    floats = [np.zeros(m, dtype=np.float64) for _ in range(5)]
    _kernels.call("tree_tables", "python", left, right, n, *ints, *floats)
    ints[3] = ints[3][:n]  # order: leaf ids in left-to-right dendrogram order
    return Tree(n, left, right, *ints, *floats)


class _CutNodes(NamedTuple):
    nodes: np.ndarray
    labels: np.ndarray
    value: float


def cut_from_nodes(tree: Tree, nodes, scores: np.ndarray) -> _CutNodes:
    """Validate that ``nodes`` tile the leaves; return canonical nodes, labels, value."""
    n = tree.n
    nodes = np.asarray(nodes)
    if nodes.ndim != 1 or nodes.size == 0 or not np.issubdtype(nodes.dtype, np.integer):
        raise ValueError("a cut must be a non-empty 1-d integer array of node ids")
    nodes = nodes.astype(np.int64)
    if np.any(nodes < 0) or np.any(nodes >= 2 * n - 1):
        raise ValueError(f"cut node ids must lie in [0, {2 * n - 1})")
    st = tree.start[nodes]
    o = np.argsort(st, kind="stable")
    nodes, st = nodes[o], st[o]
    sz = tree.size[nodes]
    if st[0] != 0 or np.any(st[1:] != st[:-1] + sz[:-1]) or st[-1] + sz[-1] != n:
        raise ValueError("cut nodes must be disjoint subtrees that cover every leaf")
    canon = np.argsort(tree.minleaf[nodes], kind="stable")
    cluster = np.empty(len(nodes), dtype=np.int64)
    cluster[canon] = np.arange(len(nodes))
    labels = np.empty(n, dtype=np.int64)
    labels[tree.order] = np.repeat(cluster, sz)
    nodes = nodes[canon]
    return _CutNodes(nodes, labels, math.fsum(scores[nodes]))
