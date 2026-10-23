"""Adaptive cut of node dendrograms of networks, with modularity as objective.

Node dendrograms come from scikit-network (Paris, Louvain hierarchy) and from
scipy (Ward linkage on a spectral embedding). On each dendrogram the
modularity is maximised by the best single-level cut and by the adaptive cut
sampler with default settings (``T="auto"``, starting from the single-level
cut); Louvain (flat) is the reference. Graphs with planted
communities of different sizes and densities (LFR benchmark, varying-density
SBM) give ground truth for ARI/AMI.

Run with ``python benchmarks/networks.py``; results go to
``benchmarks/results/networks.csv``. Needs networkx, scikit-network, sklearn.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker process

import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import adaptive_cut as ac

HERE = Path(__file__).resolve().parent


def _csr(A):
    """Symmetric 0/1 csr_matrix with int32 indices, the format scikit-network expects."""
    A = sp.csr_matrix(A, dtype=float)
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    return A


def lfr(mu, seed):
    import networkx as nx

    for s in range(seed * 100, seed * 100 + 50):
        try:
            G = nx.LFR_benchmark_graph(
                1000,
                2.5,
                1.5,
                mu,
                average_degree=12,
                max_degree=60,
                min_community=15,
                max_community=300,
                seed=s,
            )
        except nx.ExceededMaxIterations:
            continue
        comms = {frozenset(G.nodes[v]["community"]) for v in G}
        y = np.empty(G.number_of_nodes(), int)
        for c, nodes in enumerate(comms):
            y[list(nodes)] = c
        A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), format="csr")
        A.setdiag(0)
        A.eliminate_zeros()
        return _csr(A > 0), y
    raise RuntimeError("LFR generation failed")


def varying_sbm(seed):
    rng = np.random.default_rng(seed)
    sizes = np.array([200, 150, 100, 60, 40, 30, 20])
    p_in = np.array([0.04, 0.06, 0.1, 0.15, 0.25, 0.35, 0.5])
    y = np.repeat(np.arange(len(sizes)), sizes)
    i, j = np.triu_indices(len(y), 1)
    p = np.where(y[i] == y[j], p_in[y[i]], 0.004)
    keep = rng.random(i.size) < p
    n = len(y)
    A = sp.coo_array((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n)).tocsr()
    return _csr(A + A.T), y


def dendrograms(A, seed):
    from scipy.cluster.hierarchy import linkage
    from sknetwork.embedding import Spectral
    from sknetwork.hierarchy import LouvainHierarchy, Paris

    out = {"paris": Paris().fit_predict(A)}
    Z = LouvainHierarchy(random_state=seed).fit_predict(A).copy()
    n = A.shape[0]
    size = np.ones(2 * n - 1)
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        size[n + i] = size[a] + size[b]
    Z[:, 3] = size[n:]  # scikit-network 0.33 writes wrong cluster sizes here
    out["louvain_hierarchy"] = Z
    emb = Spectral(n_components=16).fit_transform(A)
    out["spectral_ward"] = linkage(emb, "ward")
    return out


def evaluate(task):
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.metrics import adjusted_rand_score as ari
    from sknetwork.clustering import Louvain, get_modularity

    warnings.filterwarnings("ignore")
    graph, param, seed = task
    A, y = lfr(param, seed) if graph == "lfr" else varying_sbm(seed)
    louv = Louvain(random_state=seed).fit_predict(A)
    base = dict(
        graph=graph,
        param=param,
        seed=seed,
        n=A.shape[0],
        m=A.nnz // 2,
        k_true=len(np.unique(y)),
        louvain_Q=get_modularity(A, louv),
        louvain_k=len(np.unique(louv)),
        louvain_ari=ari(y, louv),
        louvain_ami=ami(y, louv),
    )
    rows = []
    for name, Z in dendrograms(A, seed).items():
        scores = ac.modularity_scores(Z, A)
        lv = ac.best_level_cut(Z, scores)
        best = ac.adaptive_cut(Z, scores, "auto", rng=seed).best
        assert abs(best.value - get_modularity(A, best.labels)) < 1e-9
        rows.append(
            dict(
                base,
                dendrogram=name,
                B=ac.balancedness(Z),
                level_Q=lv.value,
                level_k=lv.n_clusters,
                level_ari=ari(y, lv.labels),
                level_ami=ami(y, lv.labels),
                adaptive_Q=best.value,
                adaptive_k=best.n_clusters,
                adaptive_ari=ari(y, best.labels),
                adaptive_ami=ami(y, best.labels),
            )
        )
    return rows


def main():
    import pandas as pd

    tasks = [("lfr", mu, s) for mu in (0.1, 0.3, 0.5) for s in range(5)]
    tasks += [("varying_sbm", 0, s) for s in range(5)]
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        rows = [r for part in pool.map(evaluate, tasks) for r in part]
    df = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    df.to_csv(HERE / "results" / "networks.csv", index=False)
    print(f"{len(df)} rows written")


if __name__ == "__main__":
    main()
