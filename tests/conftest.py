"""Tree factories and brute-force oracles, independent of the package internals."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

ZACHARY_EDGES = np.array(
    [
        (1, 2),
        (1, 3),
        (2, 3),
        (1, 4),
        (2, 4),
        (3, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (5, 7),
        (6, 7),
        (1, 8),
        (2, 8),
        (3, 8),
        (4, 8),
        (1, 9),
        (3, 9),
        (3, 10),
        (1, 11),
        (5, 11),
        (6, 11),
        (1, 12),
        (1, 13),
        (4, 13),
        (1, 14),
        (2, 14),
        (3, 14),
        (4, 14),
        (6, 17),
        (7, 17),
        (1, 18),
        (2, 18),
        (1, 20),
        (2, 20),
        (1, 22),
        (2, 22),
        (24, 26),
        (25, 26),
        (3, 28),
        (24, 28),
        (25, 28),
        (3, 29),
        (24, 30),
        (27, 30),
        (2, 31),
        (9, 31),
        (1, 32),
        (25, 32),
        (26, 32),
        (29, 32),
        (3, 33),
        (9, 33),
        (15, 33),
        (16, 33),
        (19, 33),
        (21, 33),
        (23, 33),
        (24, 33),
        (30, 33),
        (31, 33),
        (32, 33),
        (9, 34),
        (10, 34),
        (14, 34),
        (15, 34),
        (16, 34),
        (19, 34),
        (20, 34),
        (21, 34),
        (23, 34),
        (24, 34),
        (27, 34),
        (28, 34),
        (29, 34),
        (30, 34),
        (31, 34),
        (32, 34),
        (33, 34),
    ]
)

# two triangles joined by a bridge edge
TWO_TRIANGLES = np.array([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)])


def linkage_from_rows(rows, heights):
    """Build a scipy linkage matrix from (left, right) rows and heights."""
    rows = list(rows)
    n = len(rows) + 1
    size = [1] * n
    Z = []
    for (a, b), h in zip(rows, heights):
        s = size[a] + size[b]
        size.append(s)
        Z.append([a, b, h, s])
    return np.array(Z, dtype=float)


def caterpillar(n):
    rows = [(0, 1)] + [(n + i - 1, i + 1) for i in range(1, n - 1)]
    return linkage_from_rows(rows, range(1, n))


def balanced(depth, tied=True):
    n = 2**depth
    rows, heights = [], []
    level = list(range(n))
    nxt = n
    for w in range(depth):
        new = []
        for t in range(0, len(level), 2):
            rows.append((level[t], level[t + 1]))
            heights.append(w + 1 if tied else len(rows))
            new.append(nxt)
            nxt += 1
        level = new
    return linkage_from_rows(rows, heights)


def mixed6():
    """6 leaves; node 8 has a leaf and an internal child, node 10 an internal and a leaf."""
    return linkage_from_rows([(0, 1), (3, 4), (2, 7), (6, 8), (9, 5)], [1, 1, 2, 3, 4])


def random_tied(n, n_heights, rng):
    rng = np.random.default_rng(rng)
    active = list(range(n))
    rows = []
    nxt = n
    while len(active) > 1:
        i, j = sorted(rng.choice(len(active), 2, replace=False), reverse=True)
        a, b = active.pop(i), active.pop(j)
        rows.append((a, b))
        active.append(nxt)
        nxt += 1
    heights = np.sort(rng.integers(0, n_heights, n - 1)).astype(float)
    return linkage_from_rows(rows, heights)


def children(Z, v):
    n = len(Z) + 1
    return int(Z[v - n, 0]), int(Z[v - n, 1])


def leaves_of(Z, v):
    n = len(Z) + 1
    if v < n:
        return frozenset([v])
    a, b = children(Z, v)
    return leaves_of(Z, a) | leaves_of(Z, b)


def enumerate_cuts(Z, v=None):
    """All cuts of the subtree of v (default root) as frozensets of node ids."""
    n = len(Z) + 1
    if v is None:
        v = 2 * n - 2
    if v < n:
        return [frozenset([v])]
    a, b = children(Z, v)
    return [frozenset([v])] + [
        x | y for x, y in product(enumerate_cuts(Z, a), enumerate_cuts(Z, b))
    ]


def constrained_cuts(Z, v):
    """Cuts of v's subtree other than {v} in which at least one child stays whole."""
    a, b = children(Z, v)
    return [c for c in enumerate_cuts(Z, v) if c != frozenset([v]) and (a in c or b in c)]


def entropy_terms(sizes, n):
    p = np.asarray(sizes, float) / n
    return -np.sum(p * np.log2(p))


def balancedness_oracle(Z):
    """Eq. 5 by direct evaluation of every level with fcluster."""
    from scipy.cluster.hierarchy import fcluster

    n = len(Z) + 1
    ratios = []
    for h in np.unique(Z[:, 2]):
        sizes = np.bincount(fcluster(Z, h, "distance"))[1:]
        k = len(sizes)
        q, r = divmod(n, k)
        hmax = entropy_terms([q + 1] * r + [q] * (k - r), n)
        hmin = entropy_terms([1] * (k - 1) + [n - k + 1], n)
        if hmax - hmin > 1e-12:
            ratios.append((entropy_terms(sizes, n) - hmin) / (hmax - hmin))
    return float(np.mean(ratios)) if ratios else float("nan")


def reference_transition_matrix(Z, scores, T):
    """Exact MH transition matrix of the adaptive-cut chain, by enumeration.

    Proposal: pick a cluster uniformly; leaves go up, the root goes down,
    others go up or down with probability 1/2. Up merges into the parent and
    removes every cluster under it; down replaces the cluster by a
    constrained cut drawn with probability 2^-|s| / sum(2^-|s'|).
    """
    n = len(Z) + 1
    root = 2 * n - 2
    states = enumerate_cuts(Z)
    index = {s: i for i, s in enumerate(states)}
    parent = {}
    for i in range(n - 1):
        a, b = children(Z, n + i)
        parent[a] = parent[b] = n + i
    under = {
        v: {u for u in range(2 * n - 1) if leaves_of(Z, u) <= leaves_of(Z, v)}
        for v in range(2 * n - 1)
    }
    S = len(states)
    Q = np.zeros((S, S))
    for s in states:
        k = len(s)
        for c in s:
            if c < n:
                pu, pd = 1.0, 0.0
            elif c == root:
                pu, pd = 0.0, 1.0
            else:
                pu = pd = 0.5
            if pu:
                p = parent[c]
                t = frozenset(x for x in s if x not in under[p]) | {p}
                Q[index[s], index[t]] += pu / k
            if pd:
                cc = constrained_cuts(Z, c)
                zc = sum(2.0 ** -len(x) for x in cc)
                for x in cc:
                    t = (s - {c}) | x
                    Q[index[s], index[t]] += pd / k * 2.0 ** -len(x) / zc
    f = np.array([math.fsum(scores[list(s)]) for s in states])
    P = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            if i != j and Q[i, j] > 0:
                P[i, j] = Q[i, j] * min(1.0, math.exp((f[j] - f[i]) / T) * Q[j, i] / Q[i, j])
        P[i, i] = 1.0 - P[i].sum()
    pi = np.exp((f - f.max()) / T)
    return states, f, P, pi / pi.sum()


@pytest.fixture
def zachary():
    return ZACHARY_EDGES.copy()
