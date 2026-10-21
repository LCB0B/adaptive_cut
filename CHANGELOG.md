# Changelog

## 0.1.0

First release.

- `balancedness(Z)`: balancedness of any scipy linkage matrix.
- `adaptive_cut(Z, scores, T)`: multi-level cut of any dendrogram under an objective that is additive over clusters (reversible-jump MCMC with simulated annealing), optional numba engine with identical results.
- `optimal_cut(Z, scores)`: exact best multi-level cut in O(n) by one bottom-up pass.
- `best_level_cut`, `node_scores`: single-level cut and per-node scores for custom objectives.
- `link_communities(graph)`: link clustering of networks (networkx graph, edge list or sparse adjacency), partition density, single-level and adaptive cuts, node memberships.
- Examples: networks (`examples/sbm.py`), point clouds and iris, wine and digits (`examples/points.py`), a tutorial notebook, and the README figures (`examples/make_figures.py`).
