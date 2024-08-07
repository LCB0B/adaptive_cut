import numpy as np
import math

def swap(a,b):
    if a > b:
        return b,a
    return a,b


def index_2to1(i, j, n):
    if i > j:
        i, j = j, i
    return n * i + j - ((i + 2) * (i + 1)) // 2

def index_1to2(k, n):
    i = n - 2 - int(math.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    j = (k + (i + 2) * (i + 1) // 2)%n
    return i, j

def Dc(m, n):
    try:
        return (m * (m - n + 1.0)) / ((n - 2.0) * (n - 1.0))
    except ZeroDivisionError:
        return 0.0


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


if __name__ == '__main__':
    print('not a main function')