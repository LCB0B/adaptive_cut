import numpy as np
import pytest
from conftest import ZACHARY_EDGES, caterpillar, mixed6, random_tied

from adaptive_cut import adaptive_cut, link_communities

pytest.importorskip("numba")


def _cases():
    zach = link_communities(ZACHARY_EDGES, adaptive=False)
    yield mixed6(), np.random.default_rng(0).normal(size=11), 0.5
    Z = random_tied(300, 30, 1)
    yield Z, np.random.default_rng(1).normal(size=599), 0.3
    yield caterpillar(2000), np.random.default_rng(2).normal(size=3999), 1.0
    yield zach.Z, zach.scores, 1e-4


@pytest.mark.parametrize("case", list(range(4)))
@pytest.mark.parametrize("seed", range(3))
def test_numba_equals_python(case, seed):
    Z, s, T = list(_cases())[case]
    out = {}
    for engine in ("python", "numba"):
        rng = np.random.default_rng(seed)
        r = adaptive_cut(
            Z,
            s,
            T,
            n_steps=20_000,
            init="leaves" if case == 2 else "best_level",
            rng=rng,
            engine=engine,
            trace_every=1,
        )
        out[engine] = (r, rng.bit_generator.state)
    (a, sa), (b, sb) = out["python"], out["numba"]
    assert b.engine == "numba"
    assert np.array_equal(a.trace, b.trace)
    assert np.array_equal(a.best.nodes, b.best.nodes)
    assert np.array_equal(a.final.nodes, b.final.nodes)
    assert a.n_accepted == b.n_accepted
    assert sa == sb
