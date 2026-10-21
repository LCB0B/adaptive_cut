import numpy as np
import pytest
from conftest import caterpillar, mixed6, random_tied
from scipy.cluster.hierarchy import fcluster, is_valid_linkage, leaves_list, linkage

from adaptive_cut._tree import as_linkage, build_tree, cut_from_nodes


def test_valid_scipy_linkages_accepted():
    X = np.random.default_rng(0).normal(size=(30, 2))
    for method in ["single", "complete", "average", "ward", "centroid"]:
        Z = linkage(X, method)
        Z2, n = as_linkage(Z)
        assert n == 30 and np.array_equal(Z, Z2)


@pytest.mark.parametrize("seed", range(20))
def test_rejects_invalid_linkages(seed):
    rng = np.random.default_rng(seed)
    Z = random_tied(12, 4, rng)
    bad = Z.copy()
    i = int(rng.integers(len(Z)))
    kind = seed % 4
    if kind == 0:
        bad[i, 0] = bad[(i + 1) % len(Z), 1]  # a node merged twice
    elif kind == 1:
        bad[i, 3] += 1  # wrong size
    elif kind == 2:
        bad[i, 0] = 12 + len(Z)  # refers to a non-existing node
    else:
        bad[i, 2] = -1.0
    if kind in (2, 3):
        assert not is_valid_linkage(bad)
    with pytest.raises(ValueError):
        as_linkage(bad)


def test_shape_errors():
    with pytest.raises(ValueError):
        as_linkage(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        as_linkage(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        as_linkage([[0, 1, np.nan, 2]])


@pytest.mark.parametrize("seed", range(5))
def test_leaf_order_matches_scipy(seed):
    Z = random_tied(25, 5, seed)
    tree = build_tree(*as_linkage(Z))
    assert np.array_equal(tree.order, leaves_list(Z))


def test_cut_labels_match_fcluster():
    Z = random_tied(30, 5, 1)
    tree = build_tree(*as_linkage(Z))
    scores = np.zeros(59)
    for h in np.unique(Z[:, 2]):
        f = fcluster(Z, h, "distance")
        # nodes of the level: clusters whose leaf set is an fcluster block
        n = 30
        created = np.r_[np.full(n, -1), np.arange(n - 1)]
        prow = np.full(2 * n - 1, n - 1)
        prow[Z[:, 0].astype(int)] = np.arange(n - 1)
        prow[Z[:, 1].astype(int)] = np.arange(n - 1)
        i = np.flatnonzero(Z[:, 2] == h).max()
        nodes = np.flatnonzero((created <= i) & (prow > i))
        c = cut_from_nodes(tree, nodes, scores)
        _, canon = np.unique(f, return_inverse=True)
        # same partition: labels are a relabelling of each other
        pairs = set(zip(c.labels.tolist(), canon.tolist()))
        assert len(pairs) == len(set(c.labels.tolist())) == len(set(canon.tolist()))
        assert c.labels[0] == 0
        first_seen = np.unique(c.labels, return_index=True)[1]
        assert np.all(np.diff(first_seen) > 0)  # canonical order


@pytest.mark.parametrize(
    "nodes",
    [[0, 1, 2, 3, 4], [6, 6, 2, 7, 5], [6, 8, 5, 3], [11], [-1, 10], np.array([0.0, 9.0])],
)
def test_invalid_cuts(nodes):
    tree = build_tree(*as_linkage(mixed6()))
    with pytest.raises(ValueError):
        cut_from_nodes(tree, nodes, np.zeros(11))


def test_deep_caterpillar_no_recursion():
    Z = caterpillar(100_000)
    tree = build_tree(*as_linkage(Z))
    assert np.all(np.isfinite(tree.logZ)) and np.all(np.isfinite(tree.logZc[100_000:]))
