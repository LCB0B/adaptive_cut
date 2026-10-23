"""Link communities of Zachary's karate club: single-level and adaptive cut.

Link clustering puts every edge of the network in one community, so nodes can
belong to several communities. The edges are clustered by single linkage on
the similarity of their endpoints' neighbourhoods, and each cut of the
dendrogram is scored by its partition density D. Run with
``python examples/network.py`` (needs networkx).
"""

import networkx as nx

import adaptive_cut as ac


def run():
    G = nx.karate_club_graph()
    r = ac.link_communities(G, rng=0)  # edge dendrogram, single-level and adaptive cut
    best = r.cut
    for name, cut in [("single-level cut", r.level_cut), ("adaptive cut", best)]:
        large = [c for c in r.communities(cut) if len(c) >= 4]
        print(f"{name:>16}: D = {cut.value:.3f}, {len(large)} communities of 4+ nodes")
    return G, r, best


if __name__ == "__main__":
    run()
