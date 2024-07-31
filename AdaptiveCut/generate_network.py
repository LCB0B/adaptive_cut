import networkx as nx
import numpy as np
import random

#create stochastic block model
def stochastic_block_model(n, k, p, q):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            if i//k == j//k:
                if np.random.rand() < p:
                    G.add_edge(i, j)
            else:
                if np.random.rand() < q:
                    G.add_edge(i, j)
    #return the adjacency matrix
    return nx.to_numpy_matrix(G)

def stochastic_block_model_weight(n, k, p, q):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(1, n):
            if i//k == j//k:
                if np.random.rand() <= p:
                    G.add_edge(i, j,weight=np.random.randint(1,2))
            else:
                if np.random.rand() <= q:
                    G.add_edge(i, j,weight=np.random.randint(1,2))
    #return the adjacency matrix
    return nx.to_numpy_matrix(G)

#BA network directed
def generate_directed_ba_network(n, m, p=0.5):
    """
    Generates a directed BA network with n nodes and m edges added at each step.
    The direction of each edge is chosen at random with a parameter p.

    Parameters:
    n (int): Number of nodes in the network.
    m (int): Number of edges to attach from a new node to existing nodes.
    p (float): Probability of directing an edge from a newer node to an older node.

    Returns:
    G (DiGraph): Directed BA network.
    """
    # Create an undirected BA network
    G = nx.barabasi_albert_graph(n, m)

    # Convert to directed graph
    DG = nx.DiGraph()

    # Add nodes to the directed graph
    DG.add_nodes_from(G.nodes())

    # Randomly assign directions to edges
    for u, v in G.edges():
        if random.random() < p:
            DG.add_edge(u, v)  # Add edge from u to v
        else:
            DG.add_edge(v, u)  # Add edge from v to u

    return nx.to_numpy_matrix(DG)
            


if __name__=='__main__':
    n = 10
    k = 5
    p = 0.9
    q = 0.1
    adjacency_matrix = stochastic_block_model(n, k, p, q)
    #save the edge list as csv, 2d numpy array
    edge_list = np.argwhere(adjacency_matrix).astype(int)
    np.savetxt(f'data/sbm_{n}.csv', edge_list, delimiter=',', fmt='%d')

    n = 100
    k = 20
    p = 0.9
    q = 0.05
    adjacency_matrix = stochastic_block_model_weight(n, k, p, q)
    #save the edge list as csv, 2d numpy array
    edge_list = np.argwhere(adjacency_matrix).astype(int)
    edge_list_w = [[e[0],e[1],adjacency_matrix[e[0],e[1]]] for e in edge_list]
    np.savetxt(f'data/sbm_{n}_w.csv', edge_list_w, delimiter=',', fmt='%d')
    
    n = 100
    m = 10
    adjacency_matrix = generate_directed_ba_network(n, m)
    #test number of connected components in adjacency matrix
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i,j]:
                G.add_edge(i,j)
    print(nx.number_weakly_connected_components(G))
    #save the edge list as csv, 2d numpy array
    edge_list = np.argwhere(adjacency_matrix).astype(int)
    edge_list_w = [[e[0],e[1],adjacency_matrix[e[0],e[1]]] for e in edge_list]
    np.savetxt(f'data/ba_{n}.csv', edge_list_w, delimiter=',', fmt='%d')