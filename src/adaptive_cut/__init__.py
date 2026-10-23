"""Adaptive multi-level cuts and balancedness of dendrograms.

* :func:`balancedness` -- how evenly a dendrogram splits at each level.
* :func:`adaptive_cut` -- multi-level cut of any scipy linkage matrix under
  an objective that is additive over clusters, by reversible-jump
  Metropolis-Hastings sampling with simulated annealing.
* :func:`best_level_cut` -- the best single-level cut, as a baseline.
* :func:`link_communities` -- link clustering of a network with partition
  density, single-level and adaptive cuts.
* :class:`AdaptiveClustering` -- scikit-learn style estimator: scipy linkage,
  Gaussian BIC and the adaptive cut.
* :func:`gaussian_scores`, :func:`modularity_scores` -- ready-made per-node
  objectives for point clouds and for node dendrograms of networks.
"""

from .balance import Levels, balancedness
from .cut import AdaptiveCutResult, Cut, adaptive_cut, best_level_cut, node_scores
from .estimator import AdaptiveClustering, linkage_from_sklearn
from .link import LinkCommunities, link_communities, partition_density_scores
from .objectives import gaussian_scores, modularity_scores

__version__ = "0.2.0"

__all__ = [
    "AdaptiveClustering",
    "AdaptiveCutResult",
    "Cut",
    "Levels",
    "LinkCommunities",
    "adaptive_cut",
    "balancedness",
    "best_level_cut",
    "gaussian_scores",
    "link_communities",
    "linkage_from_sklearn",
    "modularity_scores",
    "node_scores",
    "partition_density_scores",
]
