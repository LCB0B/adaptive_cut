#%%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import *
import numpy as np
import networkx as nx
from logger import logger
#from fa2 import ForceAtlas2
import sys

sys.setrecursionlimit(10000)

def dendrogram_plot(num_edges, linkage, similarity_value, orig_cid2edge, newcid2cids, cid2numedges, level, colors_dict, main_path, imgname):

    # Find leaders
    linkage_np = np.array(linkage)
    leaders = level[similarity_value] 

    # Setup colors
    D_leaf_colors = {key:"#808080" for key in cid2numedges.keys()}  

    for key,value in newcid2cids.items():
        value = list(value)
        result = value.copy()
        for item in result:
            if newcid2cids.get(item):
                result.extend(newcid2cids[item])

        if key in leaders:
            if key >= num_edges:
                for val in result:
                    D_leaf_colors[val] = colors_dict[key]

    link_cols = {}
    for i, i12 in enumerate(linkage_np[:,:2].astype(int)):
        c1, c2 = (D_leaf_colors[x] for x in i12)
        link_cols[i+1+len(linkage_np)] = c1

    # Setup labels
    labels=list('' for i in range(num_edges))
    for key, val in orig_cid2edge.items():
        labels[key] = val

    np.save(main_path+'labels_lc.npy',link_cols)

    # Create plot
    fig, ax = plt.subplots(figsize=(25,10))
    ax.axis('off')
    plt.rcParams['axes.grid'] = False
    #plt.rcParams['axes.facecolor'] = 'white'
    #plt.rcParams['axes.edgecolor'] = 'white'
    #plt.figure(figsize=(20,20))
    hierarchy.dendrogram(Z=linkage, labels=labels, link_color_func=lambda x: link_cols[x])
    plt.axhline(y=similarity_value, c='k')
    plt.savefig(main_path+imgname+'.png')
    plt.close()
    return link_cols


def plot_ratio(ratio_list,main_path,best_index):
    # Create plot
    fig, ax = plt.subplots(figsize=(15,10))
    #plt.rcParams['axes.grid'] = False
    ax.plot(ratio_list,'k.')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.vlines(x=best_index,ymin=0,ymax=1)
    plt.savefig(main_path+'ratio'+'.png')
    print('plot ratio')
    plt.close()


def dendrogram_greedy(linkage, best_partitions, cid2numedges, newcid2cids, orig_cid2edge, colors_dict, main_path, imgname):

    linkage_np = np.array(linkage)
    best_partitions = sorted(best_partitions, reverse=True)

    D_leaf_colors = {key:"#808080" for key in cid2numedges.keys()}  
    for key,value in newcid2cids.items():
        value = list(value)
        result = value.copy()
        for item in result:
            if newcid2cids.get(item):
                result.extend(newcid2cids[item])
    
        if key in best_partitions:
            for val in result:
                D_leaf_colors[val] = colors_dict[key]

    link_cols = {}
    for i, i12 in enumerate(linkage_np[:,:2].astype(int)):
        c1, c2 = (D_leaf_colors[x] for x in i12)
        link_cols[i+1+len(linkage_np)] = c1


    # Create plot
    fig, ax = plt.subplots(figsize=(25,10))
    ax.axis('off')
    plt.rcParams['axes.grid'] = False
    np.save(main_path+'labels_mc.npy',link_cols)
    hierarchy.dendrogram(Z=linkage, labels=list(orig_cid2edge.values()), link_color_func=lambda x: link_cols[x])
    fig.savefig(main_path+imgname+'.png')
    plt.close()
    return link_cols



def tuning_metrics(list_D, list_clusters, threshold, main_path, imgname1, imgname2):

    sns.set_style('darkgrid')
    sns.set_palette('pastel')

    p = sns.lineplot(x=list_D.keys(), y=list_D.values())
    p.set(title='Partition density for each iteration')
    p.set_xlabel('Iterations', fontsize=10)
    p.set_ylabel('Partition density', fontsize=10)
    plt.axvline(threshold, color='#AA4A44')
    #plt.text(threshold-5, max(list_D.values())-0.003, "Threshold", rotation='vertical', size='small', color='#AA4A44')
    plt.savefig(main_path+imgname1+'.png')
    plt.close()

    p = sns.lineplot(x=list_clusters.keys(), y=list_clusters.values())
    p.set(title='Number of clusters for each iteration')
    p.set_xlabel('Iterations', fontsize=10)
    p.set_ylabel('Number of Clusters', fontsize=10)
    plt.savefig(main_path+imgname2+'.png')
    plt.close()


# def graph_plot(partitions, part_dens, filename, delimiter, num_edges, colors_dict, cid2edges, newcid2cids, main_path):

#     fig, axes = plt.subplots(1, len(list(partitions.keys())), figsize=(40, 15))
#     fig.subplots_adjust(hspace = 0.2, wspace = 1)
#     ax = axes.flatten()
#     logger.warning(f'ax {ax}')

#     G = nx.read_edgelist(filename, delimiter=delimiter, nodetype=int)

#     forceatlas2 = ForceAtlas2(
#                         # Behavior alternatives
#                         outboundAttractionDistribution=True,  # Dissuade hubs
#                         linLogMode=False,  # NOT IMPLEMENTED
#                         adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
#                         edgeWeightInfluence=1.0,

#                         # Performance
#                         jitterTolerance=10.0,  # Tolerance
#                         barnesHutOptimize=True,
#                         barnesHutTheta=1.2,
#                         multiThreaded=False,  # NOT IMPLEMENTED

#                         # Tuning
#                         scalingRatio=100.0,
#                         strongGravityMode=False,
#                         gravity=0.2,

#                         # Log
#                         verbose=False)

#     positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)

#     i = 0

#     for method in partitions.keys():

#         edge_color = {}
#         leaders = partitions[method]

#         for leader in leaders:
#             if leader < num_edges:
#                 color = '#808080'

#                 # color map
#                 edge_color.update({edge: color for edge in cid2edges[leader]})

#             else:
#                 color = colors_dict[leader]

#                 value = newcid2cids[leader]
#                 value = list(value)
#                 result = value.copy()
#                 for item in result:
#                     if newcid2cids.get(item):
#                         result.extend(newcid2cids[item])

#                 # color map
#                 edge_color.update({edge: color for val in result for edge in cid2edges[val] if val < num_edges})
            

#         edge_color = dict(sorted(edge_color.items()))

#         nx.draw_networkx_nodes(G, positions, node_size=5, node_color="black", alpha=0.2, ax=ax[i])
#         nx.draw_networkx_edges(G, positions, edge_color=list(edge_color.values()), alpha=0.8, ax=ax[i])

#         ax[i].set_title(f'{method}, {part_dens[method]:.2f}', fontsize=24)
#         ax[i].set_axis_off()

#         i += 1

#     plt.savefig(main_path+'graphs.png')
#     plt.close()


def entropy_plot(entropy, max_entropy, main_path):

    import matplotlib.pyplot as plt
    #sns.set_style('darkgrid')
    #sns.set_palette('pastel')
    import matplotlib as mpl
    mpl.style.use('default')
    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(list(entropy.values()),list(entropy.keys()), color='black')
    plt.plot(list(max_entropy.values()), list(max_entropy.keys()),  color='dodgerblue')
    #plt.title('Entropy at each level')
    #plt.xlabel('Entropy', fontsize=10)
    plt.ylabel('Level', fontsize=10)
    fig.savefig(main_path+'entropy.png')
    plt.close()

