# %%
from itertools import combinations, chain
from collections import defaultdict
from copy import copy
from scipy.cluster import hierarchy 
import numpy as np
import math
from helper_functions import *

# Mapping from (i, j) in adjacency matrix to index in condensed distance matrix
def index_2to1(i, j, n):
    if i > j:
        i, j = j, i
    return n * i + j - ((i + 2) * (i + 1)) // 2

def index_1to2(k, n):
    i = n - 2 - int(math.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    j = (k + (i + 2) * (i + 1) // 2)%n
    return i, j
  
def Dc2(m, n):
    try:
        return ( (m - n + 1.0)) / (n*(n - 1.0)/2 -(n - 1.0))
    except ZeroDivisionError:
        return 0.0

def compute_partition_density(edges2com,inv_edges):
    ms = [len(e) for e in edges2com.values()]
    list_edges = [e for e in edges2com.values()]
    list_nodes = [ set([ node for edges in e for node in inv_edges[edges] ]) for e in list_edges ]
    ns = [len(n) for n in list_nodes]
    D = 0
    for n,m in zip(ns,ms):
        D += m*Dc2(m,n)
    return D / np.sum(ms)

def entropy(comm):
    return -np.sum([c*np.log2(c) for c in comm if c > 0])
    
class LinkClustering:
    def __init__(self, name, delimiter,file='data'):
        self.file = file
        self.name = name
        self.delimiter = delimiter
        self.filename = f'{self.file}/{self.name}.csv'
        

    def read_edgelist_unweighted(self):
        adj = defaultdict(set)
        edges = set()

        with open(self.filename) as f:
            for line in f:
                ni, nj = line.strip().split(self.delimiter)
                if ni != nj:
                    ni, nj = int(ni), int(nj)
                    edges.add(swap(ni, nj))
                    adj[ni].add(nj)
                    adj[nj].add(ni)
        edges = sorted(edges)
        edges = {e: i for i, e in enumerate(edges)  }
        self.len_edges = len(edges)
        return dict(adj), edges
    
                    
    def similarities_unweighted(self):
        inclusive = {n: self.adj[n] | {n} for n in self.adj}
        similarities = []

        for node in self.adj:
            if len(self.adj[node]) > 1:
                for i, j in combinations(self.adj[node], 2):
                    edge = swap(swap(i, node), swap(node, j))
                    inc_ni, inc_nj = inclusive[i], inclusive[j]
                    jaccard_index = len(inc_ni & inc_nj) / len(inc_ni | inc_nj)
                    similarities.append((1 - jaccard_index, (self.edges[edge[0]],self.edges[edge[1]])))

        similarities.sort(key=lambda x: (x[0], x[1]))
        return similarities
    
    def single_linkage_scipy(self):
        y = np.ones(len(self.edges)*(len(self.edges)-1)//2)
        for simi, (i, j) in self.similarities:
            y[index_2to1(i, j, len(self.edges))] = simi 
        Z = hierarchy.linkage(y, method='single',optimal_ordering=False)
        self.Z = Z


    def single_linkage(self):
        linkage = np.zeros((len(self.edges)-1, 4))
        edges2comid = {i: i for i in range(len(self.edges))}
        edges2com = {i: {i} for i in range(len(self.edges))}
        k=0
        for simi in self.similarities:
            sim, edges = simi
            i,j = edges
            #print(i,j,sim)
            #if different communities
            if edges2comid[i] != edges2comid[j]:
                #merge the two communities
                edges2com[len(self.edges)+k] = edges2com[edges2comid[i]] | edges2com[edges2comid[j]]

                #update the linkage matrix
                linkage[k] = [edges2comid[i],edges2comid[j],sim,len(edges2com[len(self.edges)+k])]
                
                #update all the edges in the two communities
                for e in edges2com[edges2comid[i]]:
                    edges2comid[e] = len(self.edges)+k
                for e in edges2com[edges2comid[j]]:
                    edges2comid[e] = len(self.edges)+k
                k+=1
            #del edges2com[edges2comid[i]], edges2com[edges2comid[j]]
            #delete the two communities
            #del edges2comid[i], edges2comid[j]
            if k == len(self.edges)-1:
                break
            #print(edges2comid)
            #print(edges2com)
            #print(linkage[k])
            #print('-----------------')
        self.linkage = linkage
    

    
    def get_partition_density(self):
        D = np.zeros(len(self.linkage)+1)
        self.inv_edges = {v: k for k, v in self.edges.items()}
        k=0
        edges2comid = {i: i for i in range(len(self.edges))}
        edges2com = {i: {i} for i in range(len(self.edges))}
        best_partition = {}
        max_D = 0
        for i,j,_,_ in self.linkage:
            #print(i,j)
            edges2com[len(self.edges)+k] = edges2com[i] | edges2com[j]
            for e in edges2com[i]:
                edges2comid[e] = len(self.edges)+k
            for e in edges2com[j]:
                edges2comid[e] = len(self.edges)+k
         
            # remove the two communities
            del edges2com[i], edges2com[j]
            #print(edges2comid, edges2com)
            #print('-----------------')
            D[k] = compute_partition_density(edges2com,self.inv_edges)
            #print(D[k],max_D)
            if D[k] > max_D:
                #print(D[k],edges2com)
                best_partition = edges2com.copy()
                max_D = D[k]
            k+=1 
        self.D = D
        self.D_lc = np.max(D)
        self.best_lc_partition = best_partition
    
    def get_balanceness(self):
        n_edges = len(self.edges) 
        edges2comid = {i: i for i in range(len(self.edges))}
        edges2com = {i: {i} for i in range(len(self.edges))}
        real_entropy = np.zeros(len(self.linkage))
        max_entropy = np.zeros(len(self.linkage))
        k=0
        for i,j,_,_ in self.linkage:
            #print(i,j)
            edges2com[len(self.edges)+k] = edges2com[i] | edges2com[j]
            for e in edges2com[i]:
                edges2comid[e] = len(self.edges)+k
            for e in edges2com[j]:
                edges2comid[e] = len(self.edges)+k
         
            # remove the two communities
            del edges2com[i], edges2com[j]
            #print(edges2comid, edges2com)
            #print('-----------------')
            #print(edges2com)
            #print([len(e)/n_edges for e in edges2com.values()])
            real_entropy[k] = entropy([len(e)/n_edges for e in edges2com.values()])
            max_entropy[k]= np.log2(len(edges2com))
            #print(real_entropy[k],max_entropy[k])
            k+=1
        #correct for same similarity
        similarity = self.linkage[:,2]
        #where sim value change
        mask_sim = np.where(similarity[:-1] != similarity[1:])

        self.real_entropy = real_entropy
        self.max_entropy = max_entropy
    
    def choose_direction(self,x,partition):
        if x < self.len_edges:
            return 'up'
        elif len(partition[x]) == len(self.edges):
            return 'down'
        else:
            return np.random.choice(['up','down'])
        

    # Function to get the leaves of a specific node
    def get_leaves(node, leaf_set=None):
        if leaf_set is None:
            leaf_set = set()
        if node.is_leaf():
            leaf_set.add(node.id)
        else:
            get_leaves(node.get_left(), leaf_set)
            get_leaves(node.get_right(), leaf_set)
        return leaf_set
    
    def get_new_partition(self,direction,partition,x):
        if direction == 'up':
            #find x in linkage
            i = np.where(self.linkage[:,:2] == x)[0][0]
            #merge the communities
            if (self.linkage[i,0] in partition) and (self.linkage[i,1] in partition): 
                partition[self.len_edges+i] = partition[self.linkage[i,0]] | partition[self.linkage[i,1]]
                del partition[self.linkage[i,0]], partition[self.linkage[i,1]]
                return partition
            else:
                return partition
        if direction == 'down':
            #find x in partition
            i,j = self.linkage[x-self.len_edges,:2].astype(int)
            partition[i] = get_leaves(self.clusters[i])
            partition[j] = get_leaves(self.clusters[j])
            del partition[x]
            return partition
            
    def adaptive_cut(self,T=0.5,C=0.5,steps=1000,early_stop=1e-2):
        # test if D from get_partition_density exists
        if not hasattr(self, 'D'):
            self.get_partition_density()
       
        self.tree, self.clusters= hierarchy.to_tree(self.linkage,rd=True)
            
        partition = self.best_lc_partition
        D = self.D_lc
        
        k=0
        while k < steps:
            #choose where to walk 
            x = list(self.best_lc_partition.keys())[np.random.choice(len(self.best_lc_partition.keys()))]
            #inversly proportional to the size of the community ???
            
            
            direction = self.choose_direction(x,partition)
            
            temp_partition = self.get_new_partition(direction,partition,x)
            
            temp_D = compute_partition_density(partition,self.inv_edges)
            
            alpha = min(np.exp((temp_D-D)/T),1)
            print(f'alpha: {alpha}, temp_D: {temp_D}, D: {D}')  
            if np.random.rand() < alpha:
                partition = temp_partition
                D = temp_D
            k+=1
        
        self.partition_mcmc = partition
        self.D_mcmc = D
        
        
    def single_linkage_HC(self):
        linkage = []
        D = 0.0
        list_D = [(0.0, 1.0)]
        list_D_plot = [(0.0, 0.0)]
        S_prev = -1.0
        M = 2 / num_edges
        newcid2cids = {}

        cid2numedges_tmp = copy(cid2numedges)
        cid2numnodes_tmp = copy(cid2numnodes)

        for oms, edges in chain(similarities, [(1.0, (None, None))]):
            sim = 1 - oms
            if sim != S_prev:
                list_D.append((D, sim))
                list_D_plot.append((D, oms))
                S_prev = sim

            edge1, edge2 = edges[0], edges[1]
            if not edge1 or not edge2:
                continue

            comm_id1, comm_id2 = edge2cid[edge1], edge2cid[edge2]
            if comm_id1 == comm_id2:
                continue

            m1, m2 = len(cid2edges[comm_id1]), len(cid2edges[comm_id2])
            n1, n2 = len(cid2nodes[comm_id1]), len(cid2nodes[comm_id2])
            Dc1, Dc2 = Dc(m1, n1), Dc(m2, n2)

            if m2 > m1:
                comm_id1, comm_id2 = comm_id2, comm_id1

            curr_maxcid += 1
            newcid = curr_maxcid
            newcid2cids[newcid] = swap(comm_id1, comm_id2)
            cid2edges[newcid] = cid2edges[comm_id1] | cid2edges[comm_id2]
            cid2nodes[newcid] = set()

            for e in chain(cid2edges[comm_id1], cid2edges[comm_id2]):
                cid2nodes[newcid] |= set(e)
                edge2cid[e] = newcid

            del cid2edges[comm_id1], cid2edges[comm_id2]
            del cid2nodes[comm_id1], cid2nodes[comm_id2]

            m, n = len(cid2edges[newcid]), len(cid2nodes[newcid])
            cid2numedges_tmp[newcid] = m
            cid2numnodes_tmp[newcid] = n

            linkage.append((comm_id1, comm_id2, oms, m))

            Dc12 = Dc(m, n)
            D += (Dc12 - Dc1 - Dc2) * M

        return linkage, list_D_plot, newcid2cids, cid2numedges_tmp, cid2numnodes_tmp
        
    def build_dendrogram(self):
        self.adj, self.edges = self.read_edgelist_unweighted()
        self.similarities = self.similarities_unweighted()
        #self.linkage = self.single_linkage_scipy() 
        self.linkage = self.single_linkage()
        return 