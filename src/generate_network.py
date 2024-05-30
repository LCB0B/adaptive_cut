import networkx as nx
import numpy as np

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

if __name__=='__main__':
    n = 2000
    k = 100
    p = 0.8
    q = 0.2
    adjacency_matrix = stochastic_block_model(n, k, p, q)
    #save the edge list as csv, 2d numpy array
    edge_list = np.argwhere(adjacency_matrix).astype(int)
    np.savetxt(f'data/sbm_{n}.csv', edge_list, delimiter=',', fmt='%d')
    
    