import numpy as np
import networkx as nx

from plots import plot_dendrogram as pl 
import time

import matplotlib.pyplot as plt

name = 'sbm_10'
name = 'unbalanced_example'

from src import LinkClustering as lc
dendrogram = lc.LinkClustering(name, delimiter=',')

dendrogram = LinkClustering(name, delimiter=',')

dendrogram.build_dendrogram()

dendrogram.single_linkage()
linkage = dendrogram.linkage

dendrogram.get_partition_density()

dendrogram.D_lc
dendrogram.D
dendrogram.best_lc_partition

dendrogram.adaptive_cut(T=3e-2,steps=1e4)
dendrogram.get_balanceness()

#test all edges are ordered
for i,j in dendrogram.edges:
    assert i < j
#test len similarities is (n 2) with n len(edges)
assert len(dendrogram.similarities) == len(dendrogram.edges)*(len(dendrogram.edges)-1)//2

start_time = time.time()
pl.dendrogram_plot(len(edges), linkage, 0.5, orig_cid2edge, newcid2cids, cid2numedges, similarities,main_path='figures/', imgname=filename.split('/')[-1].split('.')[0])
print(f'dendrogram_plot: {time.time() - start_time:.2f}s')

dendrogram_plot(len(edges), linkage, 0.5, orig_cid2edge, newcid2cids, cid2numedges, similarities,main_path='figures/', imgname=filename.split('/')[-1].split('.')[0])