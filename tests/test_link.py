import math

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import TWO_TRIANGLES, ZACHARY_EDGES

from adaptive_cut import link_communities, node_scores, partition_density_scores
from adaptive_cut.link import _canonical_edges, _edge_pairs, _partition_density_temperature


def _random_graph(seed, n=30, p=0.2):
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(n, 1)
    keep = rng.random(i.size) < p
    return np.column_stack([i[keep], j[keep]]), rng


def _brute_similarity(edges, w=None):
    N = edges.max() + 1
    W = np.zeros((N, N))
    ww = np.ones(len(edges)) if w is None else w
    W[edges[:, 0], edges[:, 1]] = ww
    W[edges[:, 1], edges[:, 0]] = ww
    deg = (W > 0).sum(1)
    for x in range(N):
        if deg[x]:
            W[x, x] = W[x].sum() / deg[x]
    out = {}
    for a in range(len(edges)):
        for b in range(a + 1, len(edges)):
            sa, sb = set(edges[a].tolist()), set(edges[b].tolist())
            shared = sa & sb
            if len(shared) != 1:
                continue
            i, j = (sa - shared).pop(), (sb - shared).pop()
            dot = W[i] @ W[j]
            out[(a, b)] = 1 - dot / (W[i] @ W[i] + W[j] @ W[j] - dot)
    return out


@pytest.mark.parametrize("weighted", [False, True])
def test_similarities_match_brute_force(weighted):
    E, rng = _random_graph(0)
    w = rng.uniform(0.5, 3, len(E)) if weighted else None
    edges, w = _canonical_edges(E[:, 0], E[:, 1], w)
    ea, eb, dist = _edge_pairs(edges, edges.max() + 1, w)
    got = {(a, b): x for a, b, x in zip(ea.tolist(), eb.tolist(), dist.tolist())}
    ref = _brute_similarity(edges, w)
    assert got.keys() == ref.keys()
    for key in ref:
        assert got[key] == pytest.approx(ref[key], abs=1e-12)


def test_jaccard_exact_and_unit_weights():
    E, _ = _random_graph(1)
    edges, _ = _canonical_edges(E[:, 0], E[:, 1], None)
    N = edges.max() + 1
    _, _, d0 = _edge_pairs(edges, N, None)
    _, _, d1 = _edge_pairs(edges, N, np.ones(len(edges)))
    _, _, d2 = _edge_pairs(edges, N, np.full(len(edges), 2.0))
    assert np.array_equal(d0, d1) and np.array_equal(d0, d2)


def test_zachary_anchors(zachary):
    r = link_communities(zachary, n_steps=10_000, T=1e-4, rng=0, engine="python")
    assert r.level_cut.value == pytest.approx(0.2847578347578347, rel=1e-12)
    assert r.level_cut.n_clusters == 22
    assert r.balancedness == pytest.approx(0.63077535936780, rel=1e-12)
    assert r.Z.min(axis=0)[2] == pytest.approx(2 / 7)
    assert r.adaptive.value >= 0.38
    assert r.partition_density == r.adaptive.best.value
    assert len(r.edges) == 78 and len(r.node_labels) == 34


def test_zachary_reaches_the_optimum(zachary):
    best = max(
        link_communities(zachary, n_steps=10_000, T=1e-4, rng=s, engine="python").adaptive.value
        for s in range(3)
    )
    assert best == pytest.approx(0.402991452991453, rel=1e-9)


def test_auto_temperature_rule(zachary):
    r = link_communities(zachary, n_steps=10, rng=0)
    assert r.T == 1e-3  # 0.1/78 clipped to 1e-3
    assert _partition_density_temperature(1000, 0.5) == pytest.approx(1e-4)
    assert _partition_density_temperature(1000, 0.8) == pytest.approx(3e-5)
    assert _partition_density_temperature(1000, 0.001) == pytest.approx(1e-6)
    assert _partition_density_temperature(10**9, 0.5) == 1e-8


def test_small_networks():
    tri = link_communities(np.array([(0, 1), (1, 2), (0, 2)]), adaptive=False)
    assert tri.level_cut.value == pytest.approx(1.0)
    r = link_communities(TWO_TRIANGLES, n_steps=2000, rng=0, engine="python")
    assert r.level_cut.value == pytest.approx(6 / 7, rel=1e-12)
    assert r.level_cut.n_clusters == 3
    assert r.balancedness == pytest.approx(0.86778409384362, rel=1e-12)
    assert r.adaptive.value == pytest.approx(6 / 7, rel=1e-12)


def test_partition_density_scores_brute_force(zachary):
    r = link_communities(zachary, adaptive=False)
    E, M = r.edges, len(r.edges)

    def D(leaves):
        m = len(leaves)
        n = len(np.unique(E[leaves]))
        return 0.0 if n <= 2 else 2 * m * (m - n + 1) / ((n - 1) * (n - 2)) / M

    assert np.allclose(partition_density_scores(r.Z, E), node_scores(r.Z, D), rtol=1e-14, atol=0)
    with pytest.raises(ValueError):
        partition_density_scores(r.Z, E[:-1])


def _invariants(r):
    return r.balancedness, r.level_cut.value, r.level_cut.n_clusters


def test_input_types_agree():
    nx = pytest.importorskip("networkx")
    E = ZACHARY_EDGES
    ref = link_communities(E, adaptive=False)
    G = nx.Graph()
    G.add_edges_from(E[::-1].tolist())  # insertion order differs from sorted order
    g = link_communities(G, adaptive=False)
    assert np.array_equal(g.Z, ref.Z) and np.array_equal(g.node_labels, ref.node_labels)
    N = E.max() + 1
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(N, N))
    s = link_communities(A + A.T, adaptive=False)
    assert _invariants(s) == pytest.approx(_invariants(ref))
    rev = link_communities(E[::-1, ::-1], adaptive=False)
    assert np.array_equal(rev.Z, ref.Z)
    strs = link_communities(E.astype(str), adaptive=False)  # labels sort differently
    assert _invariants(strs) == pytest.approx(_invariants(ref), rel=1e-12)
    assert set(strs.node_labels.tolist()) == {str(x) for x in range(1, 35)}


def test_weighted_inputs_agree():
    nx = pytest.importorskip("networkx")
    E, rng = _random_graph(2)
    w = rng.uniform(1, 5, len(E))
    a = link_communities(E, weights=w, adaptive=False)
    G = nx.Graph()
    G.add_weighted_edges_from([(x, y, z) for (x, y), z in zip(E.tolist(), w)])
    b = link_communities(G, weighted=True, adaptive=False)
    N = E.max() + 1
    A = sp.coo_matrix((w, (E[:, 0], E[:, 1])), shape=(N, N)).tocsr()
    c = link_communities(A + A.T, weighted=True, adaptive=False)
    assert np.array_equal(a.Z, b.Z) and np.array_equal(a.Z, c.Z)
    assert not np.array_equal(a.Z, link_communities(E, adaptive=False).Z)


def test_cleaning_and_errors():
    E = np.array([(0, 1), (1, 0), (1, 2), (2, 2), (0, 2), (2, 3)])
    r = link_communities(E, adaptive=False)
    assert r.edges.tolist() == [[0, 1], [0, 2], [1, 2], [2, 3]]
    with pytest.raises(ValueError, match="different weights"):
        link_communities([(0, 1), (1, 0), (1, 2)], weights=[1, 2, 1])
    with pytest.raises(ValueError, match="positive"):
        link_communities([(0, 1), (1, 2)], weights=[1, 0])
    with pytest.raises(ValueError, match="weights="):
        link_communities([(0, 1), (1, 2)], weighted=True)
    with pytest.raises(ValueError, match="shape"):
        link_communities(np.array([(0, 1, 1.0), (1, 2, 1.0)]))
    with pytest.raises(ValueError, match="at least 2"):
        link_communities([(0, 1), (1, 1)])
    with pytest.raises(ValueError, match="symmetric"):
        link_communities(sp.csr_matrix(np.triu(np.ones((4, 4)), 1)))


def test_networkx_errors():
    nx = pytest.importorskip("networkx")
    with pytest.raises(ValueError, match="directed"):
        link_communities(nx.DiGraph([(0, 1), (1, 2)]))
    with pytest.raises(ValueError, match="multigraph"):
        link_communities(nx.MultiGraph([(0, 1), (1, 2)]))


def test_memberships_and_communities():
    nx = pytest.importorskip("networkx")
    G = nx.karate_club_graph()
    G.add_node("isolated")
    r = link_communities(G, n_steps=2000, rng=0, engine="python")
    Mb = r.memberships()
    assert Mb.shape == (35, r.cut.n_clusters)
    rows = np.asarray(Mb.sum(axis=1)).ravel()
    iso = list(r.node_labels).index("isolated")
    assert np.allclose(np.delete(rows, iso), 1.0) and rows[iso] == 0
    comms = r.communities()
    assert len(comms) == r.cut.n_clusters
    assert set().union(*[set(c.tolist()) for c in comms]) == set(range(34))
    assert sum(len(c) for c in comms) >= 34  # link communities overlap


def test_disconnected_and_weighted_single_orientation():
    E = np.array([(0, 1), (1, 2), (0, 2), (5, 6), (6, 7), (5, 7), (8, 9)])
    r = link_communities(E, weights=np.arange(1, 8), n_steps=1000, rng=0, engine="python")
    assert r.level_cut.value == pytest.approx(math.fsum(r.scores[r.level_cut.nodes]))
    assert r.Z[-1, 2] == 1.0
    assert r.level_cut.n_clusters == 3


def test_uniform_weights_match_unweighted():
    nx = pytest.importorskip("networkx")
    base = link_communities(ZACHARY_EDGES, adaptive=False)
    for w in (0.1, 0.3, 1 / 3, 2.2, 1e200, 1e-200):
        r = link_communities(ZACHARY_EDGES, weights=np.full(78, w), adaptive=False)
        assert np.array_equal(r.Z, base.Z)
    K = nx.complete_graph(20)
    nx.set_edge_attributes(K, 0.1, "weight")
    r = link_communities(K, weighted=True, adaptive=False)
    assert np.all(r.Z[:, 2] == 0) and r.level_cut.value == pytest.approx(1.0)


def test_weighted_twins_do_not_go_negative():
    # nodes 0 and 1 have identical weighted neighbourhoods
    E = np.array([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (3, 4)])
    W = np.array([0.3, 0.7, 0.7, 0.1, 0.1, 0.9, 0.2])
    edges, w = _canonical_edges(E[:, 0], E[:, 1], W)
    _, _, dist = _edge_pairs(edges, 5, w)
    assert np.all(dist >= 0) and np.all(dist <= 1)
    link_communities(E, weights=W, n_steps=100, rng=0)


def test_mixed_and_tuple_labels():
    r = link_communities([[1, "1"], [1, 2], [2, 3], [3, 1]], adaptive=False)
    assert len(r.node_labels) == 4 and len(r.edges) == 4
    r = link_communities(
        [(("a", 1), ("b", 2)), (("b", 2), ("c", 3, 4)), (("a", 1), ("c", 3, 4))], adaptive=False
    )
    assert sorted(map(len, r.node_labels)) == [2, 2, 3]
    nx = pytest.importorskip("networkx")
    G = nx.Graph([(1, "a"), ("a", 2.5), (2.5, (0, 0)), ((0, 0), 1)])
    assert len(link_communities(G, adaptive=False).node_labels) == 4
