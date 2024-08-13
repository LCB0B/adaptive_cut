# %%
from itertools import combinations, chain
from collections import defaultdict
from copy import copy
from scipy.cluster import hierarchy 
import numpy as np
import math
import sys
import time 
from tqdm import tqdm
import pickle

from .utils import *

sys.setrecursionlimit(int(1e4))
# Mapping from (i, j) in adjacency matrix to index in condensed distance matrix
    
class LinkClustering:
    def __init__(self, fname, delimiter=None):
        """Initialize the LinkClustering class with the dataset's name and delimiter.

        Parameters:
        name (str): Name of the dataset.
        delimiter (str): Delimiter used in the dataset file.
        file (str): Directory where the dataset file is located. Default is 'data'.
        """
        #get name from path 
        self.name = fname.split('/')[-1].split('.')[0]
        if delimiter is None:
            delimiter = ' '
            print('Delimiter not specified, using space as default.')
        self.delimiter = delimiter
        self.filename = f'{fname}'
        
        #test to see if utils is imported
        try:
            swap(1,2)
        except:
            print('AdaptiveCut.utils.py not imported!!')

    def run(self,weighted=False,directed=False,adaptive_cut=False,T=1e-4,steps=1e4):
        """
        Run the LinkClustering algorithm on the dataset.

        Parameters:
        weighted (bool): True if the network is weighted, otherwise False. Default is False.
        directed (bool): True if the network is directed, otherwise False. Default is False.
        """
        if weighted and directed:
            self.read_edgelist_weighted_directed()
            self.similarities_weighted_directed()
        elif directed and not weighted:
            print('doesnt supporte directed and unweighted, reading as undirected')
            self.read_edgelist_unweighted()
            self.similarities_unweighted()
        elif weighted:
            self.read_edgelist_weighted()
            self.similarities_weighted()
        else:
            self.read_edgelist_unweighted()
            self.similarities_unweighted_h(sampling_exponent=0)
        
        #flag issue if empty similarities
        if len(self.edges) == 0:
            print('Problem! Empty edge list!')
            return
          
        self.single_linkage_legacy()
        
        print(f'test linkage: {self.test_single_linkage()}')
        
        if adaptive_cut:
            self.adaptive_cut(T,steps)
        
        self.balanceness = self.get_balanceness()
        
    def read_edgelist_unweighted(self):
        """
        Reads an unweighted edge list from a CSV file and stores it as an adjacency list.

        Outputs:
        - self.adj (dict): Dictionary mapping each node to a set of its neighbors.
        - self.edges (dict): Dictionary mapping edges (tuple of nodes) to an index.
        - self.len_edges (int): Total number of edges.
        - self.inv_edges (dict): Dictionary mapping edge indices back to edge tuples.
        """
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
        edges = {e: i for i, e in enumerate(edges)}
        self.len_edges = len(edges)
        self.adj = dict(adj)
        self.edges = edges
        self.inv_edges = {v: k for k, v in self.edges.items()}

    def read_edgelist_weighted(self):
        """
        Reads a weighted edge list from a CSV file and stores it as an adjacency list with weights.

        Outputs:
        - self.adj (dict): Dictionary mapping each node to a set of its neighbors.
        - self.edges (dict): Dictionary mapping edges (tuple of nodes) to an index.
        - self.len_edges (int): Total number of edges.
        - self.inv_edges (dict): Dictionary mapping edge indices back to edge tuples.
        - self.weight (dict): Dictionary mapping edges to their weights.
        """
        adj = defaultdict(set)
        edges = set()
        wij_dict = {}

        with open(self.filename) as f:
            for line in f:
                ni, nj, wij = line.strip().split(self.delimiter)
                ni, nj = int(ni), int(nj)
                wij = float(wij)
                if ni != nj:
                    edges.add(swap(ni, nj))
                    wij_dict[(ni, nj)] = wij
                    adj[ni].add(nj)
                    adj[nj].add(ni)
        edges = sorted(edges)
        edges = {e: i for i, e in enumerate(edges)}
        self.len_edges = len(edges)
        self.adj = dict(adj)
        self.edges = edges
        self.inv_edges = {v: k for k, v in self.edges.items()}
        self.weight = wij_dict

    def read_edgelist_weighted_directed(self):
        """
        Reads a weighted directed edge list from a CSV file and stores it as an adjacency list with weights.

        Outputs:
        - self.adj (dict): Dictionary mapping each node to a set of nodes pointing to it.
        - self.edges (dict): Dictionary mapping directed edges (tuple of nodes) to an index.
        - self.len_edges (int): Total number of edges.
        - self.inv_edges (dict): Dictionary mapping edge indices back to edge tuples.
        - self.weight (dict): Dictionary mapping edges to their weights.
        """
        adj = defaultdict(set)
        edges = set()
        wij_dict = {}

        with open(self.filename) as f:
            for line in f:
                ni, nj, wij = line.strip().split(self.delimiter)
                ni, nj = int(ni), int(nj)
                wij = float(wij)
                if ni != nj:
                    edges.add((ni, nj))
                    wij_dict[(ni, nj)] = wij
                    adj[nj].add(ni)  # Order of i and j is inverted for the adjacency matrix

        self.adj = dict(adj)

        # Ensure all nodes appear as keys in the adjacency list
        set_keys = set(self.adj.keys())
        set_values = set([item for sublist in self.adj.values() for item in sublist])
        
        diff = set_values - set_keys
        for d in diff:
            adj[d] = {d}
            wij_dict[(d, d)] = 0
            edges.add((d, d))
        
        edges = sorted(edges)
        edges = {e: i for i, e in enumerate(edges)}
        self.len_edges = len(edges)
        self.edges = edges
        self.inv_edges = {v: k for k, v in self.edges.items()}
        self.weight = wij_dict
        self.adj = dict(adj)

    def similarities_unweighted(self):
        """
        Calculates Jaccard similarities between edges in an unweighted network.

        Outputs:
        - self.similarities (list): List of tuples (1 - Jaccard index, (edge index 1, edge index 2)).
        """
        inclusive = {n: self.adj[n] | {n} for n in self.adj}
        similarities = []

        for node in self.adj:
            if len(self.adj[node]) > 1:
                for i, j in combinations(self.adj[node], 2):
                    edge = swap(swap(i, node), swap(node, j))
                    inc_ni, inc_nj = inclusive[i], inclusive[j]
                    jaccard_index = len(inc_ni & inc_nj) / len(inc_ni | inc_nj)
                    similarities.append((1 - jaccard_index, (self.edges[edge[0]], self.edges[edge[1]])))
        similarities.sort(key=lambda x: (x[0], x[1]))
        self.similarities = similarities

    def similarities_unweighted_h(self, sampling_exponent=0):
        """
        Calculates Jaccard similarities with hierarchical sampling for unweighted networks.

        Parameters:
        sampling_exponent (float): Exponent controlling the probability of sampling pairs. Default is 0.

        Outputs:
        - self.similarities (list): List of tuples (1 - Jaccard index, (edge index 1, edge index 2)).
        """
        inclusive = {n: self.adj[n] | {n} for n in self.adj}
        similarities = []

        for node in self.adj:
            if len(self.adj[node]) > 1:
                sampling_prob = len(self.adj[node]) ** (-sampling_exponent)
                all_pairs = list(combinations(self.adj[node], 2))
                num_pairs_to_sample = max(1, int(len(all_pairs) * sampling_prob))
                sampled_pairs = np.random.choice(len(all_pairs), num_pairs_to_sample, replace=False)
                pairs_to_evaluate = [all_pairs[idx] for idx in sampled_pairs]
                for i, j in pairs_to_evaluate:
                    edge = swap(swap(i, node), swap(node, j))
                    inc_ni, inc_nj = inclusive[i], inclusive[j]
                    jaccard_index = len(inc_ni & inc_nj) / len(inc_ni | inc_nj)
                    similarities.append((1 - jaccard_index, (self.edges[edge[0]], self.edges[edge[1]])))
        # Check if some edges are missing
        unique_edges = set([e[0] for _, e in similarities]).union(set([e[1] for _, e in similarities]))
        if len(unique_edges) != len(self.edges):
            print('Some edges are missing')
        similarities.sort(key=lambda x: (x[0], x[1]))
        self.similarities = similarities
        
    def similarities_weighted(self,sampling_exponent=0):
        """
        Calculates cosine similarities for weighted networks.

        Parameters:
        sampling_exponent (float): Exponent controlling the probability of sampling pairs. Default is 0.

        Outputs:
        - self.similarities (list): List of tuples (1 - cosine similarity, (edge index 1, edge index 2)).
        """
        inclusive = {n: self.adj[n] | {n} for n in self.adj}
        similarities = []
        Aij = copy(self.weight)
        #get the average weight of each node
        A = {node: sum(self.weight[(node,i)] for i in self.adj[node]) / len(self.adj[node]) for node in self.adj}
        #add to Aij the self loop
        Aij.update({(node,node): A[node] for node in self.adj})
        #pre compute the square sum of Aij
        a_sqrd = {node: sum(Aij[(node,i)]**2 for i in inclusive[node]) for node in self.adj}
        
        for node in self.adj:
            if len(self.adj[node]) > 1:
                sampling_prob= len(self.adj[node]) ** (-sampling_exponent)
                all_pairs = list(combinations(self.adj[node], 2))
                num_pairs_to_sample = max(1, int(len(all_pairs) * sampling_prob))
                sampled_pairs = np.random.choice(len(all_pairs), num_pairs_to_sample, replace=False)
                pairs_to_evaluate = [all_pairs[idx] for idx in sampled_pairs]
                for i, j in pairs_to_evaluate:
                    edges = swap(swap(i, node), swap(node, j))
                    inc_ni, inc_nj = inclusive[i], inclusive[j]
                    ai_dot_aj = sum(Aij[swap(i, k)] * Aij[swap(j, k)] for k in inc_ni & inc_nj)
                    S = ai_dot_aj / (a_sqrd[i] + a_sqrd[j] - ai_dot_aj)
                    similarities.append((1 - S, edges))

        similarities.sort(key=lambda x: (x[0], x[1]))
        #apply edges dict to similarities
        similarities = [(sim, (self.edges[e[0]],self.edges[e[1]])) for sim, e in similarities]
        self.similarities = similarities
    

    def similarities_weighted_directed(self,sampling_exponent=0):
        """
        Calculates cosine similarities for weighted and directed networks.

        Parameters:
        sampling_exponent (float): Exponent controlling the probability of sampling pairs. Default is 0.

        Outputs:
        - self.similarities (list): List of tuples (1 - cosine similarity, (edge index 1, edge index 2)).
        """
        inclusive = {n: self.adj[n] | {n} for n in self.adj}
        similarities = []
        
        Aij = copy(self.weight)
        #get the average weight of each node
        A = {node: sum(self.weight[(i,node)] for i in self.adj[node]) / len(self.adj[node]) for node in self.adj}
        #add to Aij the self loop
        Aij.update({(node,node): A[node] for node in self.adj})
        #pre compute the square sum of Aij
        a_sqrd = {node: sum(Aij[(i,node)]**2 for i in inclusive[node]) for node in self.adj}
        
        for node in self.adj:
            if len(self.adj[node]) > 1:
                sampling_prob= len(self.adj[node]) ** (-sampling_exponent)
                all_pairs = list(combinations(self.adj[node], 2))
                num_pairs_to_sample = max(1, int(len(all_pairs) * sampling_prob))
                sampled_pairs = np.random.choice(len(all_pairs), num_pairs_to_sample, replace=False)
                pairs_to_evaluate = [all_pairs[idx] for idx in sampled_pairs]
                for i, j in pairs_to_evaluate:
                    edges = ((i,node),(j,node)) #dont swap for directed
                    inc_ni, inc_nj = inclusive[i], inclusive[j]
                    ai_dot_aj = sum(Aij[(k,i)] * Aij[(k,j)] for k in inc_ni & inc_nj)
                    
                    if (a_sqrd[i] + a_sqrd[j] - ai_dot_aj) == 0 :
                        S=0
                    else:
                        S = ai_dot_aj / (a_sqrd[i] + a_sqrd[j] - ai_dot_aj)
                    if (i,node) in self.edges and (j,node) in self.edges:
                        similarities.append((1 - S, edges))
                    if (i,node) in self.edges and (node,j) in self.edges:
                        edges = ((i ,node),(node,j))
                        similarities.append((1 - S, edges))
        similarities.sort(key=lambda x: (x[0], x[1]))
        similarities = [(sim, (self.edges[e[0]],self.edges[e[1]])) for sim, e in similarities]
        self.similarities = similarities
        
    
    def single_linkage_scipy(self):
        y = np.ones(len(self.edges)*(len(self.edges)-1)//2)
        for simi, (i, j) in self.similarities:
            y[index_2to1(i, j, len(self.edges))] = simi 
        Z = hierarchy.linkage(y, method='single',optimal_ordering=False)
        self.linkage = Z

    
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
        #start_time = time.time()
        #convert to scipy format
        #reordered_linkage = hierarchy.optimal_leaf_ordering(linkage, self.similarities)
        #print("--- %s seconds ---" % (time.time() - start_time))
        self.linkage = linkage
    
    def single_linkage_legacy(self):
        """
        Legacy implementation of single-linkage hierarchical clustering.

        Outputs:
        - self.linkage (ndarray): Linkage matrix representing the hierarchical clustering.
        - self.partition_lc (dict): Best partition of the edges based on the clustering.
        - self.D_lc_max (float): Maximum partition density.
        """
        edges = list(self.edges.keys())
        edge2cid = {edge: cid for cid, edge in enumerate(edges)}
        cid2edges = {cid: {edge} for cid, edge in enumerate(edges)}
        orig_cid2edge = {cid: edge for cid, edge in enumerate(edges)}
        cid2nodes = {cid: set(edge) for cid, edge in enumerate(edges)}
        cid2numedges = {cid: 1 for cid in range(len(edges))}
        cid2numnodes = {cid: len(edge) for cid, edge in enumerate(edges)}
        best_partition = {cid:{edge} for cid, edge in enumerate(edges)}

        curr_maxcid = len(edges) - 1
        linkage = []
        D = 0.0
        list_D = [(0.0, 1.0)]
        list_D_plot = [(0.0, 0.0)]
        S_prev = -1.0
        M = 2 / self.len_edges
        newcid2cids = {}

        cid2numedges_tmp = cid2numedges.copy()
        cid2numnodes_tmp = cid2numnodes.copy()
        
       
        inv_edges = {v: k for k, v in self.edges.items()}   

        linkage = np.zeros((len(self.edges)-1, 4))
        k=0
        for oms, edge_ids in tqdm(chain(self.similarities, [(1.0, (None, None))]) ,total=len(self.similarities)+1):
            sim = 1 - oms
            if sim != S_prev:
                list_D.append((D, sim))
                list_D_plot.append((D, oms))
                S_prev = sim
                if D > np.max([d for d, _ in list_D[:-1]]):
                    best_partition = cid2edges.copy()
            edges = [inv_edges[e] for e in edge_ids if e is not None]
            if edges==[]:
                #print('empty edges')
                continue
            edge1, edge2 = edges[0], edges[1]

            #print(edge1, edge2)
            comm_id1, comm_id2 = edge2cid[edge1], edge2cid[edge2]
            #print(comm_id1, comm_id2)
            
            if comm_id1 == comm_id2:
                #print('same community')
                continue

            m1, m2 = len(cid2edges[comm_id1]), len(cid2edges[comm_id2])
            n1, n2 = len(cid2nodes[comm_id1]), len(cid2nodes[comm_id2])
            Dc_1, Dc_2 = Dc(m1, n1), Dc(m2, n2)

            if m2 > m1:
                comm_id1, comm_id2 = comm_id2, comm_id1

            curr_maxcid += 1
            newcid = curr_maxcid
            newcid2cids[newcid] = (comm_id1, comm_id2)
            cid2edges[newcid] = cid2edges.pop(comm_id1) | cid2edges.pop(comm_id2)
            cid2nodes[newcid] = cid2nodes.pop(comm_id1) | cid2nodes.pop(comm_id2)

            for e in cid2edges[newcid]:
                edge2cid[e] = newcid

            m, n = len(cid2edges[newcid]), len(cid2nodes[newcid])
            cid2numedges_tmp[newcid] = m
            cid2numnodes_tmp[newcid] = n

            linkage[k]=[comm_id1, comm_id2, oms, m]
            k+=1 
            Dc_12 = Dc(m, n)
      
            D += (Dc_12 - Dc_1 - Dc_2) * M
            

        #if the linkage is not complete
        if k < len(self.edges)-1:
            print('Linkage not complete, fill with zeros')
            #link all the remaining communities
            for i in range(k,len(self.edges)-1):
                #take two communities at random and merge them at level 1 (max)
                comm_id1, comm_id2 = np.random.choice(list(set(edge2cid.values())),2,replace=False)
                
                #normal linkaging
                m1, m2 = len(cid2edges[comm_id1]), len(cid2edges[comm_id2])
                n1, n2 = len(cid2nodes[comm_id1]), len(cid2nodes[comm_id2])
                Dc_1, Dc_2 = Dc(m1, n1), Dc(m2, n2)
                if m2 > m1:
                    comm_id1, comm_id2 = comm_id2, comm_id1
                curr_maxcid += 1
                newcid = curr_maxcid
                newcid2cids[newcid] = (comm_id1, comm_id2)
                cid2edges[newcid] = cid2edges.pop(comm_id1) | cid2edges.pop(comm_id2)
                cid2nodes[newcid] = cid2nodes.pop(comm_id1) | cid2nodes.pop(comm_id2)
                for e in cid2edges[newcid]:
                    edge2cid[e] = newcid
            
                m, n = len(cid2edges[newcid]), len(cid2nodes[newcid])
                cid2numedges_tmp[newcid] = m
                cid2numnodes_tmp[newcid] = n
                linkage[i]=[comm_id1, comm_id2, 1.0, m]
                Dc_12 = Dc(m, n)
                D += (Dc_12 - Dc_1 - Dc_2) * M
                
        self.linkage = linkage
        self.D_lc_max= np.max([d for d, _ in list_D])
        #apply self.edges to the set of values in the best partition
        self.partition_lc = {k: {self.edges[e] for e in v} for k,v in best_partition.items()}
        #self.D_lc = LinkClustering.get_partition_density(self.partition_lc,self.inv_edges)
        self.list_D = list_D

    def test_single_linkage(self):
        """
        Tests if the generated linkage matrix is valid according to SciPy's standards.

        Outputs:
        - bool: Returns True if the linkage matrix is valid, otherwise False.
        """
        return hierarchy.is_valid_linkage(self.linkage)
    
    def get_partition_density_lc(self):
        """
        Computes the partition density for the linkage clustering (LC) method.

        Outputs:
        - self.D (ndarray): Array of partition densities at each level of the dendrogram.
        - self.D_lc_max (float): Maximum partition density found.
        - self.best_lc_partition (dict): Best partition of the edges based on maximum partition density.
        """
        D = np.zeros(len(self.linkage)+1)
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
        self.D_lc_max = np.max(D)
        self.best_lc_partition = best_partition
    
    def find_similarities(self,edge):
        """
        Finds the similarities of a specific edge with all other edges.

        Parameters:
        edge (tuple): The edge for which similarities are to be found.

        Outputs:
        - list: List of similarities with other edges.
        """
        #find the similarities of an edge with all the other edges
        return [[sim, (i,j)]for sim, (i,j) in self.similarities if i == edge or j == edge]
    
    
    def get_partition_density(partition, inv_edges):
        """
        Calculates the partition density for a given partition.

        Parameters:
        partition (dict): The partition of edges to communities.
        inv_edges (dict): Inverse mapping from edge indices to edge tuples.

        Outputs:
        - D (dict): Partition density for each community.
        """
        D = {k: 0 for k in partition.keys()}
        for key, edges in partition.items():
            ms = len(edges)
            ns = len(set([node for e in edges for node in inv_edges[e]]))
            D[key] = ms * Dc2(ms, ns)
        return D

    def update_partition_density(prev_partition, prev_D, partition, inv_edges):
        """
        Updates the partition density based on changes in the partition.

        Parameters:
        prev_partition (dict): Previous partition of edges.
        prev_D (dict): Previous partition densities.
        partition (dict): New partition of edges.
        inv_edges (dict): Inverse mapping from edge indices to edge tuples.

        Outputs:
        - D (dict): Updated partition densities.
        """
        diff_key = set(prev_partition.keys()) - set(partition.keys())
        D = prev_D.copy()
        for key in diff_key:
            key = int(key)
            del D[key]
        new_key = set(partition.keys()) - set(prev_partition.keys())
        for key in new_key:
            key = int(key)
            ms = len(partition[key])
            ns = len(set([node for edges in partition[key] for node in inv_edges[edges]]))
            D[key] = ms * Dc2(ms, ns)
        return D

    def get_fast_partition_density(self):
        """
        Quickly computes the partition density using a fast update mechanism.

        Outputs:
        - self.D_lc_max (float): Maximum partition density found.
        - self.D_lc (dict): Partition densities at the best partition.
        - self.partition_lc (dict): Best partition of edges.
        """
        self.inv_edges = {v: k for k, v in self.edges.items()}
        edges2comid = {i: i for i in range(len(self.edges))}
        edges2com = {i: {i} for i in range(len(self.edges))}
        D = {k: 0 for k, v in edges2com.items()}
        list_D = np.zeros(len(self.linkage) + 1)
        k = 0
        max_D = 0

        for i, j, _, _ in self.linkage:
            i = int(i)
            j = int(j)
            prev_edges2com = edges2com.copy()
            edges2com[len(self.edges) + k] = edges2com[i] | edges2com[j]
            for e in edges2com[i]:
                edges2comid[e] = len(self.edges) + k
            for e in edges2com[j]:
                edges2comid[e] = len(self.edges) + k
            del edges2com[i], edges2com[j]
            D = LinkClustering.update_partition_density(prev_edges2com, D, edges2com, self.inv_edges)
            list_D[k] = np.sum(list(D.values())) / self.len_edges
            if list_D[k] > max_D:
                self.partition_lc = edges2com.copy()
                self.D_lc = D.copy()
                max_D = list_D[k]
            k += 1

        self.D_lc_max = np.max(list_D)

    def get_leaves_partition(self, partition):
        """
        Returns a dictionary of partition IDs to sets of leaf nodes.

        Parameters:
        partition (dict): A partition of edges.

        Outputs:
        - leaves_partition (dict): A dictionary where keys are partition IDs and values are sets of leaf nodes.
        """
        leaves_partition = {}
        for key in partition.keys():
            leaves = get_leaves(self.clusters[key])
            leaves_partition[key] = leaves
        return leaves_partition

    def get_levels_entropy(self):
        """
        Computes entropy at different levels of the dendrogram.

        Outputs:
        - self.real_entropy (ndarray): Real entropy values at different levels.
        - self.max_entropy (ndarray): Maximum possible entropy at different levels.
        - self.min_entropy (ndarray): Minimum possible entropy at different levels.
        - self.entropy_levels (ndarray): Similarity values at the respective entropy levels.
        """
        n_edges = len(self.edges)
        edges2com = {i: {i} for i in range(len(self.edges))}
        similarity = self.linkage[:, 2]
        n_level = np.sum(similarity[:-1] != similarity[1:])
        real_entropy = np.zeros(n_level + 1)
        max_entropy = np.zeros(n_level + 1)
        min_entropy = np.zeros(n_level + 1)
        c = 0
        sim_prev = 0
        com2merge = set()
        fake_com = set()

        for k, l in enumerate(self.linkage):
            i, j, sim = l[:3]
            i = int(i)
            j = int(j)
            if sim == sim_prev:
                com2merge.add(i)
                com2merge.add(j)
                fake_com.add(len(self.edges) + k)
                sim_prev = sim
                continue
            elif sim != sim_prev:
                com2merge.add(i)
                com2merge.add(j)
                com2merge = com2merge - fake_com
                edges2com[len(self.edges) + k] = set().union(*[edges2com[i] for i in com2merge])
                for i in com2merge:
                    del edges2com[i]
                com2merge = set()

                real_entropy[c] = entropy([len(e) / n_edges for e in edges2com.values()])
                #max_entropy[c] = np.log2(len(edges2com))
                max_entropy[c] =  compute_max_entropy(len(edges2com),n_edges)
    
                min_entropy[c] = -(1 - (len(edges2com) - 1) / n_edges) * np.log2((1 - (len(edges2com) - 1) / n_edges)) - (len(edges2com) - 1) / n_edges * np.log2(1 / n_edges)

                c += 1
                sim_prev = sim

        mask_sim = np.where(similarity[:-1] != similarity[1:])[0]
        mask_sim = np.append(mask_sim, len(similarity) - 1)
        self.real_entropy = real_entropy
        self.max_entropy = max_entropy
        self.min_entropy = min_entropy
        self.entropy_levels = similarity[mask_sim]

    def get_balanceness(self):
        """
        Computes the balanceness of the network based on entropy levels.

        Outputs:
        - self.balanceness (float): The balanceness value of the network.
        """
        if not hasattr(self, 'real_entropy'):
            self.get_levels_entropy()
        denominator = (self.max_entropy[:-1] - self.min_entropy[:-1])
        numerator = (self.real_entropy[:-1] - self.min_entropy[:-1])
        mask = (denominator != 0)
        ratio = numerator[mask] / denominator[mask]
        self.balanceness = np.mean(ratio[ratio > 0])
        
        #similarity weighted entropy, with self.entropy_levels
        similarity_steps = np.diff(self.entropy_levels)
        denominator = (self.max_entropy[:-1] - self.min_entropy[:-1]) 
        numerator = (self.real_entropy[:-1] - self.min_entropy[:-1])
        mask = (denominator != 0) * (numerator != 0)
        ratio = numerator[mask] / denominator[mask] * similarity_steps[mask]
        self.balanceness_weighted = np.sum(ratio[ratio > 0]) / np.sum(similarity_steps[mask][ratio > 0])
        


    def choose_direction(self,x,partition):
        """
        Chooses the direction for dendrogram traversal.

        Parameters:
        x (int): Index of the current cluster.
        partition (dict): Current partition of edges.

        Outputs:
        - str: 'up' to merge, 'down' to split, based on the current state and the end of the dendrogram.
        """
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
            LinkClustering.get_leaves(node.get_left(), leaf_set)
            LinkClustering.get_leaves(node.get_right(), leaf_set)
        return leaf_set
    
    def remove_merged_com(self,partition,leaves):
        """
        Removes merged communities from the partition.

        Parameters:
        partition (dict): Current partition of edges.
        leaves (set): Set of leaves to remove.

        Outputs:
        - dict: Updated partition with merged communities removed.
        """
        com2del = []
        for key in partition.keys():
            if not partition[key].isdisjoint(leaves):
                com2del.append(key)
        for key in com2del:
            del partition[key]
        return partition
  
    def get_new_partition(self,direction,partition,x):
        if direction == 'up':
            #find x in linkagez̄
            pos = np.where(self.linkage[:,:2] == x)
            i = pos[0][0]
            j = 1-pos[1][0] #the other one
            #merge the communities
            if (self.linkage[i,0] not in partition) or (self.linkage[i,1] not in partition):
                leaves = LinkClustering.get_leaves(self.clusters[self.linkage[i,j].astype(int)])
                #delete the other community that contains any of the leaves
                partition = self.remove_merged_com(partition,leaves)
                partition[self.linkage[i,j].astype(int)] = leaves
                #partition = get_new_partition('up',partition,x)
                #return partition
            if (self.linkage[i,0] in partition) and (self.linkage[i,1] in partition): 
                partition[self.len_edges+i] = partition[self.linkage[i,0]] | partition[self.linkage[i,1]]
                del partition[self.linkage[i,0]], partition[self.linkage[i,1]]
                return partition
            else:
                print('Error:community nor in partition')
                return partition
        if direction == 'down':
            #find x in partition
            i,j = self.linkage[x-self.len_edges,:2].astype(int)
            partition[i] = LinkClustering.get_leaves(self.clusters[i])
            partition[j] = LinkClustering.get_leaves(self.clusters[j])
            # leaf =  LinkClustering.get_leaves(self.clusters[i]) 
            # partition[i] = leaf
            # partition[j] = partition[x] - leaf
            del partition[x]
            return partition
            
    def adaptive_cut(self,T=0.5,C=0.5,steps=1000,early_stop=1e-2):
        """
        Performs an adaptive cut on the dendrogram to find a meaningful partition.

        Parameters:
        T (float): Temperature parameter for simulated annealing. Default is 0.5.
        C (float): Cooling rate. Default is 0.5.
        steps (int): Number of steps to perform. Default is 1000.
        early_stop (float): Threshold for early stopping. Default is 1e-2.

        Outputs:
        - self.partition_mcmc (dict): Best partition found during the adaptive cut.
        - self.D_mcmc (dict): Partition densities of the best partition.
        - self.D_mcmc_max (float): Maximum partition density achieved.
        """
        # test if D from get_partition_density exists
        #if not hasattr(self, 'partiton_lc'):
            #self.get_partition_density()
       
        self.tree, self.clusters= hierarchy.to_tree(self.linkage,rd=True)
            
        partition = self.partition_lc
        
        D = LinkClustering.get_partition_density(partition,self.inv_edges)
        

        for k in tqdm(range(int(steps)),total=int(steps)):
            #choose where to walk 
            x = list(partition.keys())[np.random.choice(len(partition.keys()))]
            #inversly proportional to the size of the community ???
            
            direction = self.choose_direction(x,partition)
            #print(f'x: {x}, direction: {direction}')
            temp_partition = self.get_new_partition(direction,partition.copy(),x)
            if temp_partition == partition:
                continue
                #print('no change')
            #temp_D = compute_partition_density(partition,self.inv_edges)
            temp_D = LinkClustering.update_partition_density(partition,D,temp_partition,self.inv_edges)
            
            temps_D_value = np.sum(list(temp_D.values()))/self.len_edges
            D_value = np.sum(list(D.values()))/self.len_edges
            alpha = min(np.exp((temps_D_value-D_value)/T),1)
            #print(f'alpha: {alpha}, temp_D: {temps_D_value}, D: {D_value}')
            if np.random.rand() < alpha:
                partition = temp_partition
                D = temp_D

        
        self.partition_mcmc = partition
        self.D_mcmc = D
        self.D_mcmc_max = np.sum(list(D.values()))/self.len_edges
    
    def adaptive_cut_bias(self,T=0.5,C=0.5,steps=1000,early_stop=1e-2):
        self.tree, self.clusters= hierarchy.to_tree(self.linkage,rd=True)
            
        partition = self.partition_lc
        
        D = LinkClustering.get_partition_density(partition,self.inv_edges)
        

        for k in tqdm(range(int(steps)),total=int(steps)):
            #choose where to walk 
            x = list(partition.keys())[np.random.choice(len(partition.keys()))]
            #inversly proportional to the size of the community ???
            
            direction = self.choose_direction(x,partition)
            #print(f'x: {x}, direction: {direction}')
            temp_partition = self.get_new_partition(direction,partition.copy(),x)
            if temp_partition == partition:
                continue
                #print('no change')
            #temp_D = compute_partition_density(partition,self.inv_edges)
            temp_D = LinkClustering.update_partition_density(partition,D,temp_partition,self.inv_edges)
            
            temps_D_value = np.sum(list(temp_D.values()))/self.len_edges
            D_value = np.sum(list(D.values()))/self.len_edges
            alpha = min(np.exp((temps_D_value-D_value)/T),1)
            #print(f'alpha: {alpha}, temp_D: {temps_D_value}, D: {D_value}')
            if np.random.rand() < alpha:
                partition = temp_partition
                D = temp_D

        
        self.partition_mcmc = partition
        self.D_mcmc = D
        self.D_mcmc_max = np.sum(list(D.values()))/self.len_edges
        
    def get_partition_edges(self,partition):
        """
        Converts a partition of edge indices to actual edge tuples.

        Parameters:
        partition (dict): Partition of edge indices.

        Outputs:
        - dict: Partition of edge tuples.
        """
        partition_edges = {k: {self.inv_edges[e] for e in v} for k,v in partition.items()}
        return partition_edges
    
    def get_nodes_appartenence(self,partition):
        """
        Calculates the community membership percentage for each node.

        Parameters:
        partition (dict): Partition of edge indices.

        Outputs:
        - dict: Node community membership percentages.
        """
        partition_edges = self.get_partition_edges(partition)
        #for each node get % communities it belongs to, node_appartenence = {node: {community_id: %}}
        node_appartenence = defaultdict(dict)
        for com_id,edges in partition_edges.items():
            for edge in edges:
                for node in edge:
                    if com_id in node_appartenence[node]:
                        node_appartenence[node][com_id] += 1/len(self.adj[node])
                    else:
                        node_appartenence[node][com_id] = 1/len(self.adj[node])
        return dict(node_appartenence)
        
    
    def build_dendrogram(self):
        """
        Builds the dendrogram for the unweighted network.

        Outputs:
        - self.linkage: The linkage matrix of the dendrogram.
        """
        self.adj, self.edges = self.read_edgelist_unweighted()
        self.similarities = self.similarities_unweighted()
        self.single_linkage_legacy()
        return 


    def save(self, filepath):
        """Save the LinkClustering object to a file.

        Parameters:
        filepath (str): The path to the file where the object will be saved.
        
        ex:
        lc = LinkClustering('your_file.txt', delimiter=',')
        # ... (run your clustering process)
        lc.save('path_to_save.pkl')
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load a LinkClustering object from a file.

        Parameters:
        filepath (str): The path to the file from which the object will be loaded.

        Returns:
        LinkClustering: The loaded LinkClustering object.
        
        ex: 
        lc_loaded = LinkClustering.load('path_to_save.pkl')
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def get_partition_size(partition):
        """
        Calculates the size of each community in a partition.

        Parameters:
        partition (dict): Partition of edge indices.

        Outputs:
        - dict: Community sizes.
        """
        return {k: len(v) for k, v in partition.items()}
    
    @staticmethod
    def get_partition_density_of_each_partition(partition,inv_edges):
        """
        Calculates the partition density of each community in a partition.

        Parameters:
        partition (dict): Partition of edge indices.

        Outputs:
        - dict: Community partition densities.
        """
        return {k: compute_partition_density({k: v}, inv_edges) for k, v in partition.items()}

# %%

