import numpy as np
import pytest
from conftest import (
    balanced,
    balancedness_oracle,
    caterpillar,
    linkage_from_rows,
    random_tied,
)

from adaptive_cut import balancedness


@pytest.mark.parametrize("n", [5, 8, 50])
def test_caterpillar_is_zero(n):
    B = balancedness(caterpillar(n))
    assert B == pytest.approx(0.0, abs=1e-12)


def test_balanced_tied_is_one():
    assert balancedness(balanced(3, tied=True)) == pytest.approx(1.0, rel=1e-12)
    assert balancedness(balanced(5, tied=True)) == pytest.approx(1.0, rel=1e-12)


def test_balanced_untied():
    # the k = 3 level is {4, 2, 2} while the most even split is {3, 3, 2}
    B, lv = balancedness(balanced(3, tied=False), return_levels=True)
    assert B == pytest.approx(0.9754887502163468, rel=1e-12)
    assert list(lv.n_clusters) == [7, 6, 5, 4, 3, 2, 1]
    assert np.isnan(lv.ratio[0]) and np.isnan(lv.ratio[-1])


def test_k_not_dividing_n():
    Z = linkage_from_rows([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)], [1, 1, 2, 3, 4])
    assert balancedness(Z) == pytest.approx(0.9221815234275601, rel=1e-12)


def test_general_max_entropy():
    # {2,2,1} and {3,2} are the most even splits of 5 leaves
    Z = linkage_from_rows([(0, 1), (3, 4), (5, 2), (7, 6)], [1, 1, 2, 3])
    assert balancedness(Z) == pytest.approx(1.0, rel=1e-12)


def test_uninformative_is_nan():
    assert np.isnan(balancedness(caterpillar(2)))
    assert np.isnan(balancedness(caterpillar(3)))
    star = caterpillar(10)
    star[:, 2] = 1.0  # every merge at one height
    assert np.isnan(balancedness(star))


def test_non_monotone_raises():
    Z = caterpillar(5)
    Z[1, 2] = 0.5
    with pytest.raises(ValueError, match="non-decreasing"):
        balancedness(Z)


@pytest.mark.parametrize("seed", range(30))
def test_matches_fcluster_oracle(seed):
    rng = np.random.default_rng(seed)
    Z = random_tied(int(rng.integers(4, 60)), int(rng.integers(1, 12)), rng)
    B, lv = balancedness(Z, return_levels=True)
    expected = balancedness_oracle(Z)
    if np.isnan(expected):
        assert np.isnan(B)
    else:
        assert B == pytest.approx(expected, rel=1e-10, abs=1e-12)
        assert 0.0 <= B <= 1.0
    assert np.all(np.diff(lv.height) > 0)


def test_tie_binarisation_invariance():
    # the same tied merges, binarised in two different orders
    Z1 = linkage_from_rows([(0, 1), (6, 2), (3, 4), (7, 8), (9, 5)], [1, 1, 1, 2, 2])
    Z2 = linkage_from_rows([(3, 4), (0, 1), (7, 2), (6, 8), (9, 5)], [1, 1, 1, 2, 2])
    assert balancedness(Z1) == pytest.approx(balancedness(Z2), rel=1e-12)


def test_height_relabelling_invariance():
    Z = random_tied(40, 6, 3)
    Z2 = Z.copy()
    Z2[:, 2] = np.exp(Z[:, 2])  # any increasing map of the heights
    assert balancedness(Z) == pytest.approx(balancedness(Z2), rel=1e-12)
