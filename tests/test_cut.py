import math

import numpy as np
import pytest
from conftest import caterpillar, leaves_of, mixed6, random_tied
from scipy.cluster.hierarchy import fcluster

from adaptive_cut import Cut, adaptive_cut, best_level_cut, node_scores


def _random_scores(Z, seed):
    return np.random.default_rng(seed).normal(size=2 * len(Z) + 1)


def test_node_scores_sees_every_subtree():
    Z = random_tied(15, 4, 0)
    seen = []
    s = node_scores(Z, lambda leaves: seen.append(frozenset(leaves.tolist())) or len(leaves))
    assert len(seen) == 29
    for v in range(29):
        assert seen[v] == leaves_of(Z, v)
        assert s[v] == len(seen[v])


@pytest.mark.parametrize("seed", range(20))
def test_best_level_cut_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    Z = random_tied(int(rng.integers(3, 30)), int(rng.integers(1, 8)), rng)
    n = len(Z) + 1
    scores = rng.normal(size=2 * n - 1)
    node_of = {leaves_of(Z, v): v for v in range(2 * n - 1)}
    cands = [(-math.inf, list(range(n)))]
    for h in np.unique(Z[:, 2]):
        f = fcluster(Z, h, "distance")
        cands.append(
            (h, [node_of[frozenset(np.flatnonzero(f == c).tolist())] for c in np.unique(f)])
        )
    vals = [math.fsum(scores[nodes]) for _, nodes in cands]
    b = int(np.argmax(vals))
    cut = best_level_cut(Z, scores)
    assert cut.value == pytest.approx(vals[b], rel=1e-12, abs=1e-12)
    assert sorted(cut.nodes.tolist()) == sorted(cands[b][1])
    assert cut.height == cands[b][0]
    if np.isfinite(cut.height):
        f = fcluster(Z, cut.height, "distance")
        assert len(set(zip(f.tolist(), cut.labels.tolist()))) == cut.n_clusters


def _check_result(Z, scores, r):
    n = len(Z) + 1
    for c in (r.best, r.final, r.initial):
        assert isinstance(c, Cut)
        assert sorted(np.concatenate([sorted(leaves_of(Z, v)) for v in c.nodes]).tolist()) == list(
            range(n)
        )
        assert c.value == math.fsum(scores[c.nodes])
    assert r.best.value >= r.initial.value
    if len(r.trace):
        assert r.best.value >= r.trace.max() - 1e-9


@pytest.mark.parametrize("init", ["best_level", "leaves", "root"])
def test_adaptive_cut_invariants(init):
    Z = random_tied(40, 6, 2)
    scores = _random_scores(Z, 2)
    r = adaptive_cut(Z, scores, 0.5, n_steps=3000, init=init, rng=1, engine="python")
    _check_result(Z, scores, r)
    assert len(r.trace) == 3000 and r.n_steps == 3000 and r.k_sa == 2000
    assert r.value == r.best.value and r.gain == r.best.value - r.initial.value


def test_finds_the_optimum_of_a_small_tree():
    from conftest import enumerate_cuts

    Z = random_tied(9, 4, 5)
    scores = _random_scores(Z, 5)
    opt = max(math.fsum(scores[list(c)]) for c in enumerate_cuts(Z))
    r = adaptive_cut(Z, scores, 0.3, n_steps=20_000, rng=0, engine="python")
    assert r.value == pytest.approx(opt, rel=1e-12)


def test_reproducible_and_rng_types():
    Z = random_tied(30, 5, 3)
    s = _random_scores(Z, 3)
    a = adaptive_cut(Z, s, 0.2, n_steps=2000, rng=7, engine="python")
    b = adaptive_cut(Z, s, 0.2, n_steps=2000, rng=np.random.default_rng(7), engine="python")
    c = adaptive_cut(Z, s, 0.2, n_steps=2000, rng=8, engine="python")
    assert np.array_equal(a.trace, b.trace) and np.array_equal(a.nodes, b.nodes)
    assert not np.array_equal(a.trace, c.trace)


def test_chunk_size_independence():
    Z = random_tied(30, 5, 4)
    s = _random_scores(Z, 4)
    a = adaptive_cut(Z, s, 0.3, n_steps=3000, rng=0, engine="python")
    b = adaptive_cut(Z, s, 0.3, n_steps=3000, rng=0, engine="python", _chunk=40)
    assert np.array_equal(a.trace, b.trace)
    assert np.array_equal(a.final.nodes, b.final.nodes) and a.n_accepted == b.n_accepted


def test_scale_invariance():
    Z = random_tied(30, 5, 6)
    s = np.random.default_rng(6).integers(-8, 8, size=59) / 8.0
    a = adaptive_cut(Z, s, 0.25, n_steps=3000, rng=0, engine="python")
    b = adaptive_cut(Z, 2 * s, 0.5, n_steps=3000, rng=0, engine="python")
    assert np.array_equal(2 * a.trace, b.trace)
    assert np.array_equal(a.best.nodes, b.best.nodes)


def test_init_array_and_zero_steps():
    Z = mixed6()
    s = _random_scores(Z, 0)
    r = adaptive_cut(Z, s, 1.0, n_steps=0, init=[6, 8, 5], rng=0, engine="python")
    assert sorted(r.final.nodes.tolist()) == [5, 6, 8] and len(r.trace) == 0
    assert r.best.value == r.initial.value


def test_two_leaves():
    Z = np.array([[0, 1, 1.0, 2]])
    r = adaptive_cut(Z, [1.0, 1.0, 3.0], 0.1, n_steps=200, init="leaves", rng=0, engine="python")
    assert r.nodes.tolist() == [2] and r.value == 3.0


def test_auto_temperature():
    Z = caterpillar(6)
    s = np.zeros(11)
    s[6:] = [0.1, 0.2, 0.0, 0.4, 0.0]
    r = adaptive_cut(Z, s, "auto", n_steps=10, rng=0, engine="python")
    # merge gains |s[v] - s[l] - s[r]| = 0.1, 0.1, 0.2, 0.4, 0.4 -> median 0.2
    assert r.T == pytest.approx(0.02)
    with pytest.raises(ValueError, match="constant"):
        adaptive_cut(Z, np.zeros(11), "auto")


def test_errors():
    Z = caterpillar(5)
    s = np.zeros(9)
    with pytest.raises(ValueError):
        adaptive_cut(Z, s, 0.0)
    with pytest.raises(ValueError):
        adaptive_cut(Z, s, "hot")
    with pytest.raises(ValueError):
        adaptive_cut(Z, np.zeros(8), 1.0)
    with pytest.raises(ValueError):
        adaptive_cut(Z, np.r_[np.zeros(8), np.inf], 1.0)
    with pytest.raises(ValueError):
        adaptive_cut(Z, s, 1.0, init="middle")
    with pytest.raises(ValueError):
        adaptive_cut(Z, s, 1.0, init=[5, 6])
    with pytest.raises(ValueError):
        adaptive_cut(Z, s, 1.0, engine="fortran")
    bad = Z.copy()
    bad[0, 2] = 10.0  # non-monotone
    with pytest.raises(ValueError, match="best_level"):
        adaptive_cut(bad, s, 1.0)
    adaptive_cut(bad, s, 1.0, init="leaves", n_steps=10, engine="python")


def test_missing_numba(monkeypatch):
    from adaptive_cut import _kernels

    monkeypatch.setattr(_kernels, "numba_available", lambda: False)
    monkeypatch.setitem(__import__("sys").modules, "numba", None)
    Z = caterpillar(5)
    r = adaptive_cut(Z, np.zeros(9) + 0.1, 1.0, n_steps=10, engine="auto")
    assert r.engine == "python"
    with pytest.raises(ImportError):
        adaptive_cut(Z, np.zeros(9), 1.0, n_steps=10, engine="numba")


def test_best_is_offset_invariant():
    # a large constant per leaf does not change which cut is best
    from adaptive_cut._tree import as_linkage, build_tree

    for seed in range(20):
        Z = random_tied(10, 3, seed)
        tree = build_tree(*as_linkage(Z))
        small = np.random.default_rng(seed).normal(size=19) * 1e-7
        r = adaptive_cut(
            Z, 1e6 * tree.size + small, 1e-9, n_steps=2000, init="leaves", rng=seed, engine="python"
        )
        assert r.best.value >= r.final.value
        r0 = adaptive_cut(Z, small, 1e-9, n_steps=2000, init="leaves", rng=seed, engine="python")
        assert np.array_equal(r.best.labels, r0.best.labels)


def test_best_keeps_level_height_when_not_improved():
    Z = random_tied(20, 4, 0)
    r = adaptive_cut(Z, _random_scores(Z, 0), 1e-6, n_steps=0, rng=0, engine="python")
    assert r.best is r.initial and r.best.height is not None


@pytest.mark.parametrize("seed", range(20))
def test_sampler_finds_the_best_cut_of_small_trees(seed):
    from conftest import enumerate_cuts

    rng = np.random.default_rng(seed)
    Z = random_tied(int(rng.integers(2, 10)), int(rng.integers(1, 5)), rng)
    scores = _random_scores(Z, seed)
    opt = max(math.fsum(scores[list(c)]) for c in enumerate_cuts(Z))
    r = adaptive_cut(Z, scores, "auto", n_steps=20_000, rng=seed, engine="python")
    assert r.value == pytest.approx(opt, rel=1e-12, abs=1e-12)
    assert r.value >= best_level_cut(Z, scores).value - 1e-12
