import numpy as np
import networkx as nx

from plots import plot_dendrogram as pl 
import time

import matplotlib.pyplot as plt

name = 'sbm_10'


from src import LinkClustering as lc
dendrogram = lc.LinkClustering(name, delimiter=',')

dendrogram = LinkClustering(name, delimiter=',')

dendrogram.build_dendrogram()

fig = plt.figure(figsize=(25, 10))
dn = hierarchy.dendrogram(dendrogram.Z)
plt.savefig('figures/dendrogram.png')

fig = plt.figure(figsize=(25, 10))
dn = hierarchy.dendrogram(dendrogram.Zs)
plt.savefig('figures/dendrogram_.png')

linkage = self.single_linkage()
D = self.get_partition_density()
max_entropy,real_entropy = self.get_balanceness()

#test
#test all edges are ordered
for i,j in dendrogram.edges:
    assert i < j
#test len similarities is (n 2) with n len(edges)
assert len(dendrogram.similarities) == len(dendrogram.edges)*(len(dendrogram.edges)-1)//2

start_time = time.time()
pl.dendrogram_plot(len(edges), linkage, 0.5, orig_cid2edge, newcid2cids, cid2numedges, similarities,main_path='figures/', imgname=filename.split('/')[-1].split('.')[0])
print(f'dendrogram_plot: {time.time() - start_time:.2f}s')

dendrogram_plot(len(edges), linkage, 0.5, orig_cid2edge, newcid2cids, cid2numedges, similarities,main_path='figures/', imgname=filename.split('/')[-1].split('.')[0])