"""Adaptive multi-level cuts and balancedness of dendrograms.

* :func:`balancedness` -- how evenly a dendrogram splits at each level.
* :func:`optimal_cut` -- exact best multi-level cut of any scipy linkage
  matrix under an objective that is additive over clusters.
* :func:`adaptive_cut` -- the same search by Metropolis-Hastings sampling
  with simulated annealing.
* :func:`link_communities` -- link clustering of a network with partition
  density, single-level and adaptive cuts.
"""

from .balance import Levels, balancedness
from .cut import AdaptiveCutResult, Cut, adaptive_cut, best_level_cut, node_scores, optimal_cut
from .link import LinkCommunities, link_communities, partition_density_scores

__version__ = "0.1.0"

__all__ = [
    "AdaptiveCutResult",
    "Cut",
    "Levels",
    "LinkCommunities",
    "adaptive_cut",
    "balancedness",
    "best_level_cut",
    "link_communities",
    "node_scores",
    "optimal_cut",
    "partition_density_scores",
]
