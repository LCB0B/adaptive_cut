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

def stochastic_block_model_weight(n, k, p, q):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            if i//k == j//k:
                if np.random.rand() < p:
                    G.add_edge(i, j,weight=np.random.randint(3))
            else:
                if np.random.rand() < q:
                    G.add_edge(i, j,weight=np.random.randint(3))
    #return the adjacency matrix
    return nx.to_numpy_matrix(G)

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
    p = 0.95
    q = 0.5
    adjacency_matrix = stochastic_block_model_weight(n, k, p, q)
    #save the edge list as csv, 2d numpy array
    edge_list = np.argwhere(adjacency_matrix).astype(int)
    edge_list_w = [[e[0],e[1],adjacency_matrix[e[0],e[1]]] for e in edge_list]
    np.savetxt(f'data/sbm_w{n}.csv', edge_list_w, delimiter=',', fmt='%d')
    
    