"""Adaptive cut of an arbitrary scipy dendrogram under a custom objective.

Here: nodes of a network clustered by average linkage on shortest-path
distances, with modularity as the objective. Any score that is a sum over
clusters works the same way. Run with ``python examples/any_dendrogram.py``.
"""

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import squareform

import adaptive_cut as ac


def planted_partition(sizes, p_in, p_out, seed):
    rng = np.random.default_rng(seed)
    block = np.repeat(np.arange(len(sizes)), sizes)
    i, j = np.triu_indices(len(block), 1)
    keep = rng.random(i.size) < np.where(block[i] == block[j], p_in, p_out)
    n = len(block)
    A = sp.coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n)).tocsr()
    return A + A.T


def run(seed=0):
    A = planted_partition([60, 30, 15, 8], p_in=0.3, p_out=0.01, seed=seed)
    D = shortest_path(A, unweighted=True)
    D[np.isinf(D)] = D[np.isfinite(D)].max() + 1  # in case the graph is disconnected
    Z = linkage(squareform(D, checks=False), method="average")

    degree = np.asarray(A.sum(axis=1)).ravel()
    two_m = degree.sum()

    def modularity(nodes):  # contribution of one cluster to the modularity Q
        inside = A[nodes][:, nodes].sum()
        return inside / two_m - (degree[nodes].sum() / two_m) ** 2

    scores = ac.node_scores(Z, modularity)
    level = ac.best_level_cut(Z, scores)
    result = ac.adaptive_cut(Z, scores, T="auto", rng=seed)
    print(f"single-level cut: Q={level.value:.4f} with {level.n_clusters} clusters")
    print(f"adaptive cut:     Q={result.value:.4f} with {result.n_clusters} clusters")
    return level, result


if __name__ == "__main__":
    run()
