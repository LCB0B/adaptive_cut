# %%
from itertools import combinations, chain
from collections import defaultdict
from copy import copy
from scipy.cluster import hierarchy 
import numpy as np
import math
from helper_functions import *
import sys
import time 
from tqdm import tqdm


from .utils import *

sys.setrecursionlimit(int(1e4))
# Mapping from (i, j) in adjacency matrix to index in condensed distance matrix
    
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
        self.adj = dict(adj)
        self.edges = edges
        self.inv_edges = {v: k for k, v in self.edges.items()}
 
        #return dict(adj), edges
    
    def read_edgelist_weighted(self):
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
        edges = {e: i for i, e in enumerate(edges)  }
        self.len_edges = len(edges)
        self.adj = dict(adj)
        self.edges = edges
        self.inv_edges = {v: k for k, v in self.edges.items()}
        self.weight = wij_dict
    
    def read_edgelist_weighted_directed(self):
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
                    adj[ni].add(nj)
        edges = sorted(edges)
        edges = {e: i for i, e in enumerate(edges)  }
        self.len_edges = len(edges)
        self.adj = dict(adj)
        self.edges = edges
        self.inv_edges = {v: k for k, v in self.edges.items()}
        self.weight = wij_dict
        
    
    
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
        self.similarities = similarities
        
             
    def similarities_unweighted_h(self,sampling_exponent=0):
        inclusive = {n: self.adj[n] | {n} for n in self.adj}
        similarities = []      

        for node in self.adj:
            if len(self.adj[node]) > 1:
                sampling_prob= len(self.adj[node]) ** (-sampling_exponent)
                all_pairs = list(combinations(self.adj[node], 2))
                num_pairs_to_sample = max(1, int(len(all_pairs) * sampling_prob))
                sampled_pairs = np.random.choice(len(all_pairs), num_pairs_to_sample, replace=False)
                pairs_to_evaluate = [all_pairs[idx] for idx in sampled_pairs]
                #print(f'lenght pairs to evaluate: {len(all_pairs)}, sampling prob: {sampling_prob}, n_neighbors: {len(self.adj[node])}, num_pairs_to_sample: {num_pairs_to_sample}')
                for i, j in pairs_to_evaluate:
                    edge = swap(swap(i, node), swap(node, j))
                    inc_ni, inc_nj = inclusive[i], inclusive[j]
                    jaccard_index = len(inc_ni & inc_nj) / len(inc_ni | inc_nj)
                    similarities.append((1 - jaccard_index, (self.edges[edge[0]],self.edges[edge[1]])))
        # check is some edges are missing
        unique_edges = set([e[0] for _,e in similarities]).union(set([e[1] for _,e in similarities]))
        if len(unique_edges) != len(self.edges):
            print('Some edges are missing')
        similarities.sort(key=lambda x: (x[0], x[1]))
        self.similarities = similarities
        
    def similarities_weighted(self,sampling_exponent=0):
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
                    edges = ((i,node),(j,node)) #dont swap for directed
                    inc_ni, inc_nj = inclusive[i], inclusive[j]
                    ai_dot_aj = sum(Aij[(i,k)] * Aij[(j,k)] for k in inc_ni & inc_nj)
                    S = ai_dot_aj / (a_sqrd[i] + a_sqrd[j] - ai_dot_aj)
                    similarities.append((1 - S, edges))
                    # directed network trick
                    if (node,j) in self.edges:
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
        edges = list(self.edges.keys())
        edge2cid = {edge: cid for cid, edge in enumerate(edges)}
        cid2edges = {cid: {edge} for cid, edge in enumerate(edges)}
        orig_cid2edge = {cid: edge for cid, edge in enumerate(edges)}
        cid2nodes = {cid: set(edge) for cid, edge in enumerate(edges)}
        cid2numedges = {cid: 1 for cid in range(len(edges))}
        cid2numnodes = {cid: len(edge) for cid, edge in enumerate(edges)}
        best_partition = {cid:self.edges[edge] for cid, edge in enumerate(edges)}

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
                print('empty edges')
                continue
            edge1, edge2 = edges[0], edges[1]

            print(edge1, edge2)
            comm_id1, comm_id2 = edge2cid[edge1], edge2cid[edge2]
            print(comm_id1, comm_id2)
            
            if comm_id1 == comm_id2:
                print('same community')
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
            
            # if k>100:
            #     break
            
        self.linkage = linkage
        self.D_lc_max= np.max([d for d, _ in list_D])
        #apply self.edges to the set of values in the best partition
        self.partition_lc = {k: {self.edges[e] for e in v} for k,v in best_partition.items()}
        #self.D_lc = LinkClustering.get_partition_density(self.partition_lc,self.inv_edges)
        self.list_D = list_D

    def test_single_linkage(self):
        return hierarchy.is_valid_linkage(self.linkage)
    
    def get_partition_density_lc(self):
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
        #find the similarities of an edge with all the other edges
        return [[sim, (i,j)]for sim, (i,j) in self.similarities if i == edge or j == edge]
        
    def get_partition_density(partition,inv_edges):
        #intialize the partition density with partition keys
        D={k:0 for k in partition.keys()}
        for key,edges in partition.items():
            ms = len(edges)
            ns = len(set([ node for e in edges for node in inv_edges[e] ]))
            D[key] = ms*Dc2(ms,ns)
        return D
    
    def update_partition_density(prev_partition,prev_D,partition,inv_edges):
        # prev_D : dict community_id : {edges}
        diff_key = set(prev_partition.keys()) - set(partition.keys())
        D = prev_D.copy()
        for key in diff_key:
            key = int(key)
            del D[key]
        new_key = set(partition.keys()) - set(prev_partition.keys())
        for key in new_key:
            key = int(key)
            ms = len(partition[key])
            ns = len(set([ node for edges in partition[key] for node in inv_edges[edges] ]))
            D[key] = ms*Dc2(ms,ns)
        return D
    
    def get_fast_partition_density(self):
        self.inv_edges = {v: k for k, v in self.edges.items()}
        edges2comid = {i: i for i in range(len(self.edges))}
        edges2com = {i: {i} for i in range(len(self.edges))}
        D = {k:0 for k,v in edges2com.items()}
        list_D = np.zeros(len(self.linkage)+1)
        k=0
        max_D = 0
        for i,j,_,_ in self.linkage:
            i = int(i)
            j = int(j)
            prev_edges2com = edges2com.copy()
            edges2com[len(self.edges)+k] = edges2com[i] | edges2com[j]
            for e in edges2com[i]:
                edges2comid[e] = len(self.edges)+k
            for e in edges2com[j]:
                edges2comid[e] = len(self.edges)+k
            del edges2com[i], edges2com[j]
            D = LinkClustering.update_partition_density(prev_edges2com,D,edges2com,self.inv_edges)
            list_D[k] = np.sum(list(D.values()))/self.len_edges
            if list_D[k] > max_D:
                #print(D[k],edges2com)
                self.partition_lc = edges2com.copy()
                self.D_lc = D.copy()
                max_D = list_D[k]
            k+=1
        #self.D = list_D
        self.D_lc_max = np.max(list_D)
        #return self.D_lc_max
            
    def get_leaves_partition(self,partition):
        #return a dict partition_id: set of leaves
        leaves_partition = {}
        for key in partition.keys():
            leaves = get_leaves(self.clusters[key])
            leaves_partition[key] = leaves
        #leaves_partition = {k:v for key inpartition.keys()
        return leaves_partition
                
    def get_levels_entropy(self):
        n_edges = len(self.edges) 
        edges2com = {i: {i} for i in range(len(self.edges))}
        similarity = self.linkage[:,2]
        n_level = np.sum(similarity[:-1] != similarity[1:])
        real_entropy = np.zeros(n_level+1)
        max_entropy = np.zeros(n_level+1)
        min_entropy = np.zeros(n_level+1)
        c=0
        sim_prev = 0
        com2merge = set()
        fake_com = set()
        for k,l in enumerate(self.linkage):
            i,j,sim = l[:3]
            i = int(i)
            j = int(j)
            if sim == sim_prev:
                com2merge.add(i)
                com2merge.add(j)
                fake_com.add(len(self.edges)+k)
                sim_prev = sim
                continue
            #print(i,j)
            elif sim != sim_prev: 
                com2merge.add(i)
                com2merge.add(j)
                #merge the communities
                com2merge = com2merge - fake_com
                edges2com[len(self.edges)+k] = set().union(*[edges2com[i] for i in com2merge])
                
                # remove the merged communities
                for i in com2merge:
                    del edges2com[i]
                #print(edges2comid, edges2com)
                #print('-----------------')
                #print(edges2com)
                #print([len(e)/n_edges for e in edges2com.values()])
                com2merge = set()

                real_entropy[c] = entropy([len(e)/n_edges for e in edges2com.values()])
                max_entropy[c]= np.log2(len(edges2com))
                min_entropy[c] = -(1-(len(edges2com)-1)/n_edges)*np.log2((1-(len(edges2com)-1)/n_edges)) - (len(edges2com)-1)/n_edges*np.log2(1/n_edges)
            
                #print(real_entropy[k],max_entropy[k])
                c+=1
                sim_prev = sim
        #correct for same similarity
        #where sim value change
        mask_sim = np.where(similarity[:-1] != similarity[1:])[0] # add the last similarity
        mask_sim = np.append(mask_sim,len(similarity)-1)
        self.real_entropy = real_entropy
        self.max_entropy = max_entropy
        self.min_entropy = min_entropy
        
        self.entropy_levels = similarity[mask_sim] 
        
    def balanceness(self):
        if not hasattr(self, 'real_entropy'):
            self.get_levels_entropy()
        denominator = (self.max_entropy[:-1]-self.min_entropy[:-1])
        numerator = (self.real_entropy[:-1]-self.min_entropy[:-1])
        mask = denominator != 0
        ratio = numerator[mask]/denominator[mask]
        self.balanceness = np.mean(ratio[ratio>0])

    
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
            LinkClustering.get_leaves(node.get_left(), leaf_set)
            LinkClustering.get_leaves(node.get_right(), leaf_set)
        return leaf_set
    
    def remove_merged_com(self,partition,leaves):
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
        
        
    def get_partition_edges(self,partition):
        partition_edges = {k: {self.inv_edges[e] for e in v} for k,v in partition.items()}
        return partition_edges
    
    def get_nodes_appartenence(self,partition):
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
        return node_appartenence
        
    
    def build_dendrogram(self):
        self.adj, self.edges = self.read_edgelist_unweighted()
        self.similarities = self.similarities_unweighted()
        self.single_linkage_legacy()
        return 

# %%

