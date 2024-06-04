import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import matplotlib.colors as mcolors


colors = ['#0064b0', '#9f9825', '#98d4e2', '#c04191', '#83c491', '#f3a4ba', '#ceadd2', '#d5c900', '#e3b32a', '#8d5e2a', '#00643c', '#662483', '#b90845']

def coloring_function(linkage, partition, colors):
    dflt_col = "#808080"
    leaf_partition = {leaf:pid for pid, leaves in partition.items() for leaf in leaves}
    color_dict = {pid: colors[i % len(colors)] for i, pid in enumerate(partition)}
    leaf_colors = {leaf: color_dict[pid] for leaf, pid in leaf_partition.items()}
    link_cols = {}
    for i, i12 in enumerate(linkage[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(linkage) else leaf_colors[x] for x in i12)
        link_cols[i+1+len(linkage)] = c1 if c1 == c2 else dflt_col
    return link_cols


def dendrogram_plot(ax,linkage,partition,colors):
    link_cols = coloring_function(linkage, partition, colors)
    with plt.rc_context({'lines.linewidth': 0.1}):
        hierarchy.dendrogram(linkage, 
                            ax=ax,
                            link_color_func=lambda x: link_cols[x],
                            distance_sort='descending',
                            )
    #remove x labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    #remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    

    
if __name__=='__main__':
    colors_dict = colors_dict_metro
    name = "sb"
    partition = dendrogram.partition_lc
    linkage = dendrogram.linkage
    partition = dendrogram.partition_mcmc

    
    fig, ax = plt.subplots(figsize=(10, 5))  # set size
    #hierarchy.set_link_color_palette([colors_dict[x] for x in partition])
    
    dendrogram_plot(ax,linkage,partition,colors)
    plt.savefig(f'figures/{name}_dendrogram_mcmc.png',dpi=300)
