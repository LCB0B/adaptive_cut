import numpy as np
import pytest
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, is_valid_linkage, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

from adaptive_cut._linkage import sparse_single_linkage


def naive_kruskal(n, i, j, d, fill):
    """Reference single linkage: Kruskal over (d, min, max)-sorted pairs."""
    a, b = np.minimum(i, j), np.maximum(i, j)
    parent = list(range(n))
    size = [1] * n
    cid = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    rows = []
    for t in sorted(range(len(d)), key=lambda t: (d[t], a[t], b[t])):
        x, y = find(a[t]), find(b[t])
        if x == y:
            continue
        if size[x] < size[y]:
            x, y = y, x
        rows.append([min(cid[x], cid[y]), max(cid[x], cid[y]), d[t], size[x] + size[y]])
        parent[y] = x
        size[x] += size[y]
        cid[x] = n + len(rows) - 1
    return np.array(rows)


def _random_pairs(seed, n=40, p=0.15, n_dist=5):
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(n, 1)
    keep = rng.random(i.size) < p
    i, j = i[keep], j[keep]
    d = rng.integers(0, n_dist, i.size) / n_dist  # many ties, including d = 0
    return n, i, j, d


@pytest.mark.parametrize("seed", range(10))
def test_matches_naive_kruskal_and_components(seed):
    n, i, j, d = _random_pairs(seed)
    Z = sparse_single_linkage(n, i, j, d, fill_height=1.0)
    assert is_valid_linkage(Z) and np.all(np.diff(Z[:, 2]) >= 0)
    ref = naive_kruskal(n, i, j, d, 1.0)
    assert np.array_equal(Z[: len(ref)], ref)
    for h in np.unique(d):
        keep = d <= h
        A = sp.coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
        nc, lab = connected_components(A, directed=False)
        f = fcluster(Z, h, "distance")
        assert len(set(zip(f.tolist(), lab.tolist()))) == nc == len(set(f.tolist()))


def test_matches_scipy_single_on_complete_graph():
    rng = np.random.default_rng(0)
    D = squareform(rng.integers(0, 4, size=45).astype(float))
    i, j = np.triu_indices(10, 1)
    Z = sparse_single_linkage(10, i, j, D[i, j])
    Zs = linkage(squareform(D), "single")
    assert np.array_equal(Z[:, 2], Zs[:, 2])
    for h in np.unique(Zs[:, 2]):
        a, b = fcluster(Z, h, "distance"), fcluster(Zs, h, "distance")
        assert len(set(zip(a.tolist(), b.tolist()))) == len(set(a.tolist()))


def test_zero_distances_are_kept():
    i, j = np.triu_indices(5, 1)
    Z = sparse_single_linkage(5, i, j, np.zeros(i.size))
    assert np.all(Z[:, 2] == 0) and Z[-1, 3] == 5


def test_input_order_and_duplicates():
    n, i, j, d = _random_pairs(3)
    Z = sparse_single_linkage(n, i, j, d, fill_height=1.0)
    p = np.random.default_rng(1).permutation(len(d))
    assert np.array_equal(Z, sparse_single_linkage(n, j[p], i[p], d[p], fill_height=1.0))
    # duplicate pairs keep the smallest distance
    Zd = sparse_single_linkage(n, np.r_[i, i], np.r_[j, j], np.r_[d, d + 0.5], fill_height=2.0)
    assert np.array_equal(Z[:, 2][Z[:, 2] < 1], Zd[:, 2][Zd[:, 2] < 1])


def test_forest_completion():
    # triangle {0,1,2}, edge {3,4}, isolated item 5
    Z = sparse_single_linkage(6, [0, 1, 0, 3], [1, 2, 2, 4], [0.1, 0.2, 0.3, 0.1], fill_height=7.0)
    assert is_valid_linkage(Z)
    assert list(Z[:, 2]) == [0.1, 0.1, 0.2, 7.0, 7.0]
    # components chained in order of smallest item: {0,1,2} + {3,4}, then + {5}
    assert Z[3, 3] == 5 and Z[4, 3] == 6 and 5 in Z[4, :2]


def test_no_pairs_and_errors():
    Z = sparse_single_linkage(3, [], [], [])
    assert is_valid_linkage(Z) and np.all(Z[:, 2] == 1.0)
    with pytest.raises(ValueError):
        sparse_single_linkage(1, [], [], [])
    with pytest.raises(ValueError):
        sparse_single_linkage(3, [0], [1], [-1.0])
    with pytest.raises(ValueError):
        sparse_single_linkage(3, [0], [1], [np.nan])
    with pytest.raises(ValueError):
        sparse_single_linkage(3, [0], [3], [0.5])
    with pytest.raises(ValueError):
        sparse_single_linkage(3, [0], [1], [2.0], fill_height=1.0)
