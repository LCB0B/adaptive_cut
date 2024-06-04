import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import matplotlib.colors as mcolors


colors = ['#FFCE00','#0064b0', '#9f9825', '#98d4e2', '#c04191', '#83c491', '#f3a4ba', '#ceadd2', '#d5c900', '#e3b32a', '#8d5e2a', '#00643c', '#662483', '#b90845']

def make_color_dict(colors, partition_lc, partition_mcmc):
    #merge the two partitions and remove duplicates
    partition_tot = {**partition_lc, **partition_mcmc}
    #order the partition by partition id
    partition_tot = dict(sorted(partition_tot.items()))
    #put partiton in intersection first
    partition = {pid: leaves for pid, leaves in partition_tot.items() if pid in partition_lc and pid in partition_mcmc}
    #add the rest of the partition
    partition.update({pid: leaves for pid, leaves in partition_tot.items() if pid not in partition})
    
    leaf_partition = {leaf:pid for pid, leaves in partition.items() for leaf in leaves}
    color_dict = {pid: colors[i % len(colors)] for i, pid in enumerate(partition)}
    return color_dict

def coloring_function(linkage, partition, color_dict,lim_color=1.1):
    dflt_col = "#DDDDDD"
    leaf_partition = {leaf:pid for pid, leaves in partition.items() for leaf in leaves}
    leaf_colors = {leaf: color_dict[pid] for leaf, pid in leaf_partition.items()}
    link_cols = {}
    for i, i123 in enumerate(linkage[:,:3]):
        if i123[2] > lim_color:
            link_cols[i+1+len(linkage)] = dflt_col
        else:
            i12 = i123[:2].astype(int)
            c1, c2 = (link_cols[x] if x > len(linkage) else leaf_colors[x] for x in i12)
            link_cols[i+1+len(linkage)] = c1 if c1 == c2 else dflt_col
    return link_cols


def dendrogram_plot(ax,linkage,partition,color_dict,lim_color=1.1):
    link_cols = coloring_function(linkage, partition, color_dict,lim_color)
    with plt.rc_context({'lines.linewidth': 0.5}):
        hierarchy.dendrogram(linkage, 
                            ax=ax,
                            link_color_func=lambda x: link_cols[x],
                            #distance_sort='descending',
                            )
    #remove x labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    #remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    
min_entropy = [h*1/(h) for k,h in enumerate(real_entropy)]

def plot_entropy(real_entropy,max_entropy):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(real_entropy,'k-',label='Real entropy')
    ax.plot(max_entropy,'r--',label='Max entropy')
    ax.plot(min_entropy,'b--',label='Min entropy')
    ax.plot(1,'b--',label='log2(2)')
    plt.savefig(f'figures/{name}_entropy.png')
    plt.close('all')
    
if __name__=='__main__':
    name = "balanced"
    partition_lc= dendrogram.partition_lc
    linkage = dendrogram.linkage
    partition_mcmc = dendrogram.partition_mcmc
    
    #np.random.shuffle(colors)
    color_dict = make_color_dict(colors, partition_lc, partition_mcmc)
    
    fig, axs = plt.subplots(2,1,figsize=(7, 7))  # set size
    #shuffle colors list
    dendrogram_plot(axs[0],linkage,partition_lc,color_dict,lim_color=1)
    dendrogram_plot(axs[1],linkage,partition_mcmc,color_dict)
    #plt.tight_layout()
    plt.savefig(f'figures/{name}_dendrogram_.png',dpi=300)
    
#get key intersection partition_lc and partition_mcmc

