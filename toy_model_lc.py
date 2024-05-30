import numpy as np
import networkx as nx

from src import link_clustering as lc

import time

filename = 'data/sbm.csv'

global_start_time = time.time()

start_time = time.time()
adj,edges = lc.read_edgelist_unweighted(filename, delimiter=',')
print(f'read_edgelist_unweighted: {time.time() - start_time:.2f}s')

start_time = time.time()
similarities = lc.similarities_unweighted(adj)
print(f'similarities_unweighted: {time.time() - start_time:.2f}s')

start_time = time.time()
edge2cid, cid2edges, orig_cid2edge, cid2nodes, curr_maxcid, cid2numedges, cid2numnodes = lc.initialize_edges(edges=edges)
print(f'initialize_edges: {time.time() - start_time:.2f}s')

start_time = time.time()
linkage, list_D_plot, newcid2cids, cid2numedges_m, cid2numnodes_n = lc.single_linkage_HC(
    edges=edges,
    num_edges=len(edges),
    similarities=similarities,
    edge2cid=edge2cid,
    cid2edges=cid2edges,
    cid2nodes=cid2nodes,
    curr_maxcid=curr_maxcid,
    cid2numedges=cid2numedges,
    cid2numnodes=cid2numnodes
)
print(f'single_linkage_HC: {time.time() - start_time:.2f}s')

print(f'Total time: {time.time() - global_start_time:.2f}s,  network size: {len(edges)}')