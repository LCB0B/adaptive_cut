"""Adaptive cut on a varying-density network and on a balanced network.

Run with ``python examples/sbm.py``.
"""

import numpy as np

import adaptive_cut as ac


def sbm(sizes, P, seed):
    """Edge list of a stochastic block model with block sizes and edge probabilities P."""
    rng = np.random.default_rng(seed)
    block = np.repeat(np.arange(len(sizes)), sizes)
    i, j = np.triu_indices(len(block), 1)
    keep = rng.random(i.size) < np.asarray(P)[block[i], block[j]]
    return np.column_stack([i[keep], j[keep]])


def networks(seed=0):
    # varying density: block densities 1 ... 0.1, larger blocks are sparser,
    # consecutive blocks are linked with probability p_{c+1}^2
    p = np.linspace(1, 0.1, 5)
    P = np.diag(p)
    for c in range(4):
        P[c, c + 1] = P[c + 1, c] = p[c + 1] ** 2
    varying = sbm((20 / np.sqrt(p)).astype(int), P, seed)
    # balanced: five equal blocks with the same densities
    balanced = sbm([40] * 5, 0.1 + 0.8 * np.eye(5), seed)
    return {"varying density": varying, "balanced": balanced}


def run(n_steps=50_000, seed=0):
    out = {}
    for name, edges in networks(seed).items():
        r = ac.link_communities(edges, n_steps=n_steps, rng=seed)
        out[name] = r
        print(
            f"{name:>16}: M={len(r.edges):5d}  B={r.balancedness:.2f}  "
            f"D single-level={r.level_cut.value:.3f} ({r.level_cut.n_clusters} communities)  "
            f"D adaptive={r.partition_density:.3f} ({r.cut.n_clusters} communities)"
        )
    return out


if __name__ == "__main__":
    run()
