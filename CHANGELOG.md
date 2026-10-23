# Changelog

## Unreleased

- Eight example notebooks in `examples/notebooks` (scipy linkages, scikit-learn, flat methods, HDBSCAN, link and node communities, balancedness); they replace `examples/tutorial.ipynb`.

## 0.2.0 (2026-10-23)

- `AdaptiveClustering`: scikit-learn style estimator (scipy linkage, Gaussian BIC, adaptive cut). It passes scikit-learn's `check_estimator` and works without scikit-learn installed.
- `gaussian_scores(Z, X, covariance=...)`: per-node Gaussian BIC with diagonal, spherical or full covariance, in O(n d).
- `modularity_scores(Z, A)`: per-node modularity of a node dendrogram, in O(n log n + m).
- `linkage_from_sklearn(model)`: scipy linkage matrix of a fitted `AgglomerativeClustering`.
- Benchmarks (`benchmarks/`) of the adaptive cut across scipy linkage methods and node dendrograms of networks.
- Karate club example of link communities, with a figure.
- Removed `optimal_cut`: the package provides the adaptive cut sampler only.

## 0.1.0 (2026-10-21)

First release.

- `balancedness(Z)`: balancedness of any scipy linkage matrix.
- `adaptive_cut(Z, scores, T)`: multi-level cut of any dendrogram under an objective that is additive over clusters (reversible-jump MCMC with simulated annealing), optional numba engine with identical results.
- `optimal_cut(Z, scores)`: exact best multi-level cut in O(n) by one bottom-up pass.
- `best_level_cut`, `node_scores`: single-level cut and per-node scores for custom objectives.
- `link_communities(graph)`: link clustering of networks (networkx graph, edge list or sparse adjacency), partition density, single-level and adaptive cuts, node memberships.
- Examples: networks (`examples/sbm.py`), point clouds and iris, wine and digits (`examples/points.py`), a tutorial notebook, and the README figures (`examples/make_figures.py`).
