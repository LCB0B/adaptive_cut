import math

import numpy as np
import pytest
from conftest import (
    caterpillar,
    constrained_cuts,
    enumerate_cuts,
    mixed6,
    random_tied,
    reference_transition_matrix,
)

from adaptive_cut import adaptive_cut
from adaptive_cut._tree import as_linkage, build_tree

SMALL_TREES = [
    mixed6(),
    caterpillar(5),
    random_tied(7, 3, 0),
    random_tied(7, 7, 1),
    np.array([[0, 1, 1.0, 2]]),
]


@pytest.mark.parametrize("Z", SMALL_TREES)
def test_partition_functions_match_enumeration(Z):
    n = len(Z) + 1
    tree = build_tree(*as_linkage(Z))
    for v in range(2 * n - 1):
        z = sum(2.0 ** -len(c) for c in enumerate_cuts(Z, v))
        assert math.exp(tree.logZ[v]) == pytest.approx(z, rel=1e-12)
        if v >= n:
            cc = constrained_cuts(Z, v)
            zc = sum(2.0 ** -len(c) for c in cc)
            assert math.exp(tree.logZc[v]) == pytest.approx(zc, rel=1e-12)
            assert 0 < tree.p_both[v] <= tree.p_lcum[v] <= 1 + 1e-15
            l, r = int(Z[v - n, 0]), int(Z[v - n, 1])
            both = sum(2.0 ** -len(c) for c in cc if l in c and r in c) / zc
            split_l = sum(2.0 ** -len(c) for c in cc if l not in c) / zc
            assert tree.p_both[v] == pytest.approx(both, rel=1e-12)
            assert tree.p_lcum[v] == pytest.approx(both + split_l, rel=1e-12)


@pytest.mark.parametrize("Z", SMALL_TREES[:4])
def test_reference_chain_is_reversible(Z):
    scores = np.random.default_rng(0).uniform(0, 0.5, size=2 * len(Z) + 1)
    _, _, P, pi = reference_transition_matrix(Z, scores, 0.3)
    assert np.allclose(P.sum(1), 1) and np.all(P >= -1e-15)
    flow = pi[:, None] * P
    assert np.allclose(flow, flow.T, atol=1e-14)
    assert np.allclose(pi @ P, pi, atol=1e-14)


@pytest.mark.parametrize("Z", [mixed6(), random_tied(7, 3, 0)])
def test_kernel_transitions_match_reference(Z):
    """Empirical transitions of the sampler equal the exact MH kernel."""
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 0.5, size=2 * len(Z) + 1)
    T = 0.5
    states, f, P, pi = reference_transition_matrix(Z, scores, T)
    assert np.min(np.diff(np.sort(f))) > 1e-6  # every cut is identifiable by its value
    n_steps = 200_000
    r = adaptive_cut(
        Z,
        scores,
        T,
        n_steps=n_steps,
        k_sa=n_steps,
        init="leaves",
        rng=2,
        engine="python",
        trace_every=1,
    )
    order = np.argsort(f)
    idx = np.searchsorted(f[order], r.trace - 1e-9)
    seq = order[np.minimum(idx, len(f) - 1)]
    assert np.allclose(f[seq], r.trace, atol=1e-9)
    counts = np.zeros_like(P)
    np.add.at(counts, (seq[:-1], seq[1:]), 1)
    visits = counts.sum(1)
    for i in np.flatnonzero(visits >= 2000):
        phat = counts[i] / visits[i]
        tol = 5 * np.sqrt(P[i] * (1 - P[i]) / visits[i]) + 2e-3
        assert np.all(np.abs(phat - P[i]) <= tol), (states[i], phat, P[i])
    occupancy = np.bincount(seq, minlength=len(f)) / len(seq)
    assert np.abs(occupancy - pi).max() < 0.02


# Frozen output of the Python engine: pins the order in which uniforms are
# consumed so any change to the sampler is noticed (regenerate deliberately).
GOLDEN_SCORES = np.arange(11) / 8.0
GOLDEN_NODES = [6, 2, 3, 4, 5]
GOLDEN_TRACE = [
    1.875,
    2.5,
    1.875,
    1.875,
    2.5,
    2.5,
    2.5,
    2.375,
    1.75,
    1.75,
    2.375,
    2.375,
    2.375,
    1.75,
    2.375,
    2.375,
    2.5,
    2.5,
    1.875,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    1.875,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
    2.5,
]


def test_golden_stream():
    r = adaptive_cut(
        mixed6(),
        GOLDEN_SCORES,
        0.5,
        n_steps=50,
        init="leaves",
        rng=0,
        engine="python",
        trace_every=1,
    )
    assert r.trace.tolist() == GOLDEN_TRACE
    assert r.final.nodes.tolist() == GOLDEN_NODES


def test_large_trees_run():
    for Z in (caterpillar(20_000), random_tied(5_000, 50, 3)):
        n = len(Z) + 1
        s = np.random.default_rng(0).normal(size=2 * n - 1)
        r = adaptive_cut(Z, s, 1.0, n_steps=2000, init="leaves", rng=0, engine="python")
        assert np.isfinite(r.value)
