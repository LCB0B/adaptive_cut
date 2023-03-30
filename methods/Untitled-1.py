

from itertools import permutations
from itertools import combinations, chain
from collections import defaultdict, Counter
from copy import copy
from helper_functions import *
from logger import logger
import random 


# @classmethod
# def convertIGraphToNxGraph(cls, igraph):
#     node_names = igraph.vs["name"]
#     edge_list = igraph.get_edgelist()
#     weight_list = igraph.es["weight"]
#     node_dict = defaultdict(str)

#     for idx, node in enumerate(igraph.vs):
#         node_dict[node.index] = node_names[idx]

#     convert_list = []
#     for idx in range(len(edge_list)):
#         edge = edge_list[idx]
#         new_edge = (node_dict[edge[0]], node_dict[edge[1]], weight_list[idx])
#         convert_list.append(new_edge)

#     convert_graph = nx.Graph()
#     convert_graph.add_weighted_edges_from(convert_list)
#     return convert_graph

def updateNodeWeights(edge_weights):
    node_weights = defaultdict(float)
    for node in edge_weights.keys():
        node_weights[node] = sum([weight for weight in edge_weights[node].values()])
    return node_weights


def computeModularity(node2com, edge_weights, param):
    q = 0
    all_edge_weights = get_all_edge_weights(edge_weights)
    com2node = defaultdict(list)
    for node, com_id in node2com.items():
        com2node[com_id].append(node)

    #com_id, nodes = list(com2node.items())[1]
    for com_id, nodes in com2node.items(): #com_id, nodes = list(com2node.items())[0]
        node_combinations = list(combinations(nodes, 2)) + [(node, node) for node in nodes]
        cluster_weight = sum([edge_weights[node_pair[0]][node_pair[1]] for node_pair in node_combinations if node_pair[1] in edge_weights[node_pair[0]].keys() ])
        tot = getDegreeOfCluster(nodes, node2com, edge_weights)
        q += (cluster_weight / all_edge_weights) - param * ((tot / (2 * all_edge_weights)) ** 2)
    return q

def getDegreeOfCluster( nodes, node2com, edge_weights):
    weight = sum([sum(list(edge_weights[n].values())) for n in nodes])
    self_loop_extra = sum([edge_weights[n][n] for n in nodes if n in edge_weights[n]])
    return weight +self_loop_extra

def _updatePartition(new_node2com, partition):
    reverse_partition = defaultdict(list)
    for node, com_id in partition.items():
        reverse_partition[com_id].append(node)

    for old_com_id, new_com_id in new_node2com.items():
        for old_com in reverse_partition[old_com_id]:
            partition[old_com] = new_com_id
    return partition


def get_all_edge_weights(edge_weights):
    noself = sum([weight for start in edge_weights.keys() for end, weight in edge_weights[start].items() if start!=end]) //2
    onlyself = sum([weight for start in edge_weights.keys() for end, weight in edge_weights[start].items() if start==end])
    return noself+onlyself


def _runFirstPhase(node2com, edge_weights, param,linkage,random_dict=1):
    all_edge_weights = get_all_edge_weights(edge_weights)
    #print(all_edge_weights)
    node_weights = updateNodeWeights(edge_weights)
    status = True

    #random dict order 
    if random_dict:
        l = list(node2com.items())
        random.shuffle(l)
        node2com = dict(l)

    while status:
        statuses = []
        for node in node2com.keys(): #node = list(node2com.keys())[0]
            if Counter(node2com.values())[node2com[node]]>1:
                continue
            statuses = []
            com_id = node2com[node]
            neigh_nodes = [edge[0] for edge in getNeighborNodes(node, edge_weights)]
            if node in neigh_nodes : # not going to add a node in its own community 
                neigh_nodes.remove(node) 
            max_delta = -1
            max_com_id = com_id
            communities = {} #dont try the same com twice
            for neigh_node in neigh_nodes: # neigh_node = neigh_nodes[1]
                node2com_copy = node2com.copy()
                if node2com_copy[neigh_node] in communities:
                    continue
                communities[node2com_copy[neigh_node]] = 1
                node2com_copy[node] = node2com_copy[neigh_node]
                #delta_q = 2 * getNodeWeightInCluster(node, node2com_copy, edge_weights) - (getTotWeight(node, node2com_copy, edge_weights) * node_weights[node] / all_edge_weights) * param
                delta_q1 = 1/(2*all_edge_weights) * ( getNodeWeightInCluster(node, node2com_copy, edge_weights) - (node_weights[node]*getTotWeight(node, node2com_copy, edge_weights) / all_edge_weights))
                delta_q = computeModularity(node2com_copy, edge_weights, param) - computeModularity(node2com, edge_weights, param)
                #print(delta_q1,delta_q)
                #print(delta_q,computeModularity(node2com_copy, edge_weights, param),computeModularity(node2com, edge_weights, param),computeModularity(node2com, edge_weights, param)+delta_q,computeModularity(node2com_copy, edge_weights, param)-delta_q==computeModularity(node2com, edge_weights, param))
                #print(delta_q,neigh_node)
                if delta_q > max_delta:
                    max_delta = delta_q
                    max_com_id = node2com_copy[neigh_node]
            node2com[node] = max_com_id
            if com_id != max_com_id:
                statuses.append(com_id != max_com_id)
                linkage.append((node,neigh_node,max_delta+linkage[-1][2],computeModularity(node2com, edge_weights, param)))
        if sum(statuses) == 0:
            break

    return node2com,linkage

def _runSecondPhase(node2com, edge_weights):
    com2node = defaultdict(list)

    new_node2com = {}
    new_edge_weights =defaultdict(dict)

    for node, com_id in node2com.items():
        com2node[com_id].append(node)
        if com_id not in new_node2com:
            new_node2com[com_id] = com_id

    nodes = list(node2com.keys())
    node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes] +[(node, node) for node in nodes]
    for edge in node_pairs:
        if edge[1] in edge_weights[edge[0]].keys():
            try :
                new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][edge[1]]
            except:
                new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] = edge_weights[edge[0]][edge[1]]

    #print('before ',get_all_edge_weights(new_edge_weights))

    for edge in new_edge_weights.keys():
        if edge in new_edge_weights[edge].keys():
            new_edge_weights[edge][edge] = new_edge_weights[edge][edge] //2
    
    #print('after ',get_all_edge_weights(new_edge_weights))

    return new_node2com, new_edge_weights

def getTotWeight(node, node2com, edge_weights):
    nodes = [n for n, com_id in node2com.items() if com_id == node2com[node] and node != n]

    weight = 0.
    for n in nodes:
        weight += sum(list(edge_weights[n].values()))
    return weight

def getNeighborNodes(node, edge_weights):
    if node not in edge_weights:
        return 0
    return edge_weights[node].items()

def getNodeWeightInCluster(node, node2com, edge_weights):
    neigh_nodes = getNeighborNodes(node, edge_weights)
    node_com = node2com[node]
    weights = 0.
    for neigh_node in neigh_nodes:
        if node_com == node2com[neigh_node[0]]:
            weights += neigh_node[1]
    return weights

# def _setNode2Com(self, graph):
#     node2com = {}
#     edge_weights = defaultdict(lambda : defaultdict(float))
#     for idx, node in enumerate(graph.nodes()):
#         node2com[node] = idx
#         for edge in graph[node].items():
#             edge_weights[node][edge[0]] = edge[1]["weight"]
#     return node2com, edge_weights

def _setNode2Com(dict_adj,edges):
    node2com = {}
    edge_weights = defaultdict(dict)
    for idx, node in enumerate(dict_adj.keys()):
        node2com[node] = idx
    for edge in edges:
        edge_weights[edge[0]][edge[1]] = 1 #unweighted
        edge_weights[edge[1]][edge[0]] = 1 #unweighted
    return node2com, edge_weights

def read_edgelist_unweighted(filename, delimiter=None):
    adj = defaultdict(set)
    edges = set()
    i=0
    # Loop through each edge in the network
    for line in open(filename):
        # Get the list of nodes in each edge
        L = line.strip().split(delimiter)
        ni, nj = int(L[0]), int(L[1])
        if ni != nj: 
            edges.add( swap(int(ni), int(nj)) )
            # Create adjacency dictionary
            adj[ni].add(nj)
            adj[nj].add(ni)
    return dict(adj), edges


filename = 'data/data/wiki_science.txt'
filename = 'data/data/word_adjacency-french.txt'
filename = 'data/data/Zachary.txt'

dict_adj,edges = read_edgelist_unweighted(filename, delimiter='-')
node2com, edge_weights = _setNode2Com(dict_adj,edges )


def getBestPartition(dict_adj,edges, param=1.):
    node2com, edge_weights = _setNode2Com(dict_adj,edges)
    linkage = [(0,0,computeModularity(node2com, edge_weights, param),computeModularity(node2com, edge_weights, param))]
    node2com, linkage= _runFirstPhase(node2com, edge_weights, param,linkage)
    best_modularity = computeModularity(node2com, edge_weights, param)


    partition = node2com.copy()
    new_node2com, new_edge_weights = _runSecondPhase(node2com, edge_weights)
    print(get_all_edge_weights(new_edge_weights))
    while True:
        new_node2com,linkage = _runFirstPhase(new_node2com, new_edge_weights, param,linkage) #node2com, edge_weights = new_node2com, new_edge_weights
        modularity = computeModularity(new_node2com, new_edge_weights, param)
        print(modularity)
        if (modularity-best_modularity) < 1e-3:
            break
        best_modularity = modularity
        partition = _updatePartition(new_node2com, partition)
        print(get_all_edge_weights(new_edge_weights))
        _new_node2com, _new_edge_weights = _runSecondPhase(new_node2com, new_edge_weights)
        new_node2com = _new_node2com
        new_edge_weights = _new_edge_weights
    return partition,best_modularity,linkage


def getfulllinkage(dict_adj,edges, param=1.):
    node2com, edge_weights = _setNode2Com(dict_adj,edges)
    linkage = [(0,0,computeModularity(node2com, edge_weights, param),computeModularity(node2com, edge_weights, param))]
    node2com, linkage= _runFirstPhase(node2com, edge_weights, param,linkage)
    best_modularity = computeModularity(node2com, edge_weights, param)
    partition = node2com.copy()
    new_node2com, new_edge_weights = _runSecondPhase(node2com, edge_weights)
    print(get_all_edge_weights(new_edge_weights))
    while len(new_edge_weights)>1:
        new_node2com,linkage = _runFirstPhase(new_node2com, new_edge_weights, param,linkage) #node2com, edge_weights = new_node2com, new_edge_weights
        modularity = computeModularity(new_node2com, new_edge_weights, param)
        print(modularity)
        if (modularity-best_modularity) >0:
            best_modularity = modularity
            partition = _updatePartition(new_node2com, partition)
        print(get_all_edge_weights(new_edge_weights))
        _new_node2com, _new_edge_weights = _runSecondPhase(new_node2com, new_edge_weights)
        new_node2com = _new_node2com
        new_edge_weights = _new_edge_weights
    return partition,best_modularity,linkage


dict_adj,edges = read_edgelist_unweighted(filename, delimiter='-')

partition,mod,linkage = getBestPartition(dict_adj,edges, param=1.)
partition,mod,linkage = getfulllinkage(dict_adj,edges, param=1.)
np.max([i[2]for i in linkage])
b=np.unique(list(partition.values()))
print(len(partition),len(b))
print(f'Modularity :{mod}')

import networkx as nx 
graph = nx.read_adjlist(filename,delimiter='-')
import networkx.algorithms.community as nx_comm
nx_part = nx_comm.louvain_communities(graph, seed=1)
print(f'Modularity nx:{nx_comm.modularity(graph,nx_part)}')
rr = [len(i) for i in nx_part]
from collections import defaultdict

mylouvain = defaultdict(list)

for key, value in sorted(partition.items()):
    mylouvain[str(value)].append(str(key))


myL=list(mylouvain.values())

print(f'Modularity, {nx_comm.modularity(graph,myL)},{nx_comm.modularity(graph,nx_part)}')


# computeModularity(new_node2com, new_edge_weights, param)

# mylouvain = defaultdict(list)

# for key, value in sorted(partition.items()):
#     mylouvain[str(value)].append(str(key))


# myL=list(mylouvain.values())



# c=0
# node2com = {}
# edge_weights = defaultdict(dict)
# for idx, node in enumerate(dict_adj.keys()):
#     node2com[node] = idx
# for edge in edges:
#     edge_weights[edge[0]][edge[1]] = 1 #unweighted
#     edge_weights[edge[1]][edge[0]] = 1 #unweighted
#     c+=1


# all_edge_weights = sum([weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2


# for start in edge_weights.keys() :
#     for end, weight in edge_weights[start].items():
#         print(start,end,weight)

# d = [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]
# edge_weights

# mylouvain = defaultdict(list)
# for key, value in sorted(node2com.items()):
#     mylouvain[value].append(key)

# myL=list(mylouvain.values())






from scipy.cluster.hierarchy import dendrogram, linkage

from matplotlib import pyplot as plt

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

Z = linkage(X)

fig = plt.figure(figsize=(25, 10))

dn = dendrogram(Z)

plt.show()