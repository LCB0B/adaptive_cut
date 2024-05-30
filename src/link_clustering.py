# %%
from itertools import combinations, chain
from collections import defaultdict
from copy import copy
from helper_functions import *

# Create adjacency dictionary and a set with all edges in the network
def read_edgelist_unweighted(filename, delimiter=None):
    adj = defaultdict(set)
    edges = set()

    with open(filename) as f:
        for line in f:
            ni, nj = line.strip().split(delimiter)
            if ni != nj:
                ni, nj = int(ni), int(nj)
                edges.add(swap(ni, nj))
                adj[ni].add(nj)
                adj[nj].add(ni)
    return dict(adj), edges

def read_edgelist_weighted(filename, delimiter=None):
    adj = defaultdict(set)
    edges = set()
    wij_dict = {}

    with open(filename) as f:
        for line in f:
            ni, nj, wij = line.strip().split(delimiter)
            ni, nj = int(ni), int(nj)
            wij = float(wij)
            if ni != nj:
                edges.add(swap(ni, nj))
                wij_dict[(ni, nj)] = wij
                adj[ni].add(nj)
                adj[nj].add(ni)
    return dict(adj), edges, wij_dict

# Similarities
def similarities_unweighted(adj):
    inclusive = {n: adj[n] | {n} for n in adj}
    similarities = []

    for node in adj:
        if len(adj[node]) > 1:
            for i, j in combinations(adj[node], 2):
                edges = swap(swap(i, node), swap(node, j))
                inc_ni, inc_nj = inclusive[i], inclusive[j]
                jaccard_index = len(inc_ni & inc_nj) / len(inc_ni | inc_nj)
                similarities.append((1 - jaccard_index, edges))

    similarities.sort(key=lambda x: (x[0], x[1]))
    return similarities

def similarities_weighted(adj, wij):
    inclusive = {n: adj[n] | {n} for n in adj}
    Aij = copy(wij)
    a_sqrd = {}

    for node in adj:
        Aij[(node, node)] = sum(wij[swap(node, i)] for i in adj[node]) / len(adj[node])
        a_sqrd[node] = sum(Aij[swap(node, i)]**2 for i in inclusive[node])

    similarities = []

    for node in adj:
        if len(adj[node]) > 1:
            for i, j in combinations(adj[node], 2):
                edges = swap(swap(i, node), swap(node, j))
                inc_ni, inc_nj = inclusive[i], inclusive[j]
                ai_dot_aj = sum(Aij[swap(i, k)] * Aij[swap(j, k)] for k in inc_ni & inc_nj)
                S = ai_dot_aj / (a_sqrd[i] + a_sqrd[j] - ai_dot_aj)
                similarities.append((1 - S, edges))

    similarities.sort(key=lambda x: (x[0], x[1]))
    return similarities

# Each link is initially assigned to its own community
def initialize_edges(edges):
    edge2cid = {}
    cid2edges = {}
    orig_cid2edge = {}
    cid2nodes = {}
    cid2numedges = {}
    cid2numnodes = {}

    for cid, edge in enumerate(edges):
        edge = swap(*edge)
        edge2cid[edge] = cid
        cid2edges[cid] = {edge}
        orig_cid2edge[cid] = edge
        cid2nodes[cid] = set(edge)
        cid2numedges[cid] = 1
        cid2numnodes[cid] = len(edge)

    curr_maxcid = len(edges) - 1
    return edge2cid, cid2edges, orig_cid2edge, cid2nodes, curr_maxcid, cid2numedges, cid2numnodes

# Single-linkage hierarchical clustering
def single_linkage_HC(edges, num_edges, similarities, edge2cid, cid2edges, cid2nodes, curr_maxcid, cid2numedges, cid2numnodes):
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

def link_clustering(filename, delimiter):
    adj, edges = read_edgelist_unweighted(filename=filename, delimiter=delimiter)
    similarities = similarities_unweighted(adj=adj)
    edge2cid, cid2edges, orig_cid2edge, cid2nodes, curr_maxcid, cid2numedges, cid2numnodes = initialize_edges(edges=edges)
    linkage, list_D_plot, newcid2cids, cid2numedges_m, cid2numnodes_n = single_linkage_HC(
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
    return linkage, list_D_plot, newcid2cids, orig_cid2edge, cid2numedges_m, cid2numnodes_n, len(edges), len(adj)
