import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import matplotlib.colors as mcolors

#helvetica font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5


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


def dendrogram_plot(ax,linkage,partition,color_dict,lim_color=1.1,vertical_line=False,D=0,title=''):
    link_cols = coloring_function(linkage, partition, color_dict,lim_color)
    with plt.rc_context({'lines.linewidth': 0.5}):
        hierarchy.dendrogram(linkage, 
                            ax=ax,
                            link_color_func=lambda x: link_cols[x],
                            #distance_sort='descending',
                            )
    #remove x labels
    if vertical_line:
        ax.axhline(y=lim_color, c='k',linestyle='--',linewidth=1.5,alpha=1)
    if D!=0:
        title = f'{title}, D={D:.3f}'
    ax.set_xticks([])
    ax.set_xticklabels([])
    #remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.text(0.02, 1, title, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left')
    ax.set_ylim(linkage[0][2],linkage[-1][2])

def plot_entropy(ax,real_entropy,max_entropy,min_entropy,entropy_levels,balanceness=0):
    ax.plot(real_entropy,entropy_levels,'-',label='Real entropy',color='teal')
    ax.plot(max_entropy,entropy_levels,'k--',label='Max entropy')
    ax.plot(min_entropy,entropy_levels,'k:',label='Min entropy')
    ax.fill_betweenx(entropy_levels, real_entropy, max_entropy, color='gray', alpha=0.15)
    ax.fill_betweenx(entropy_levels, min_entropy, real_entropy, color='teal', alpha=0.15)
    ax.set_xlabel('Entropy')
    #ax.set_ylabel('Similarity')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if balanceness!=0:
        ax.text(np.max(real_entropy)/2, entropy_levels[-1]/2, f'Balanceness={balanceness:.2f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.set_ylim(entropy_levels[0],entropy_levels[-1])
 
def plot_compare_methods(dendrogram_balanced,dendrogam_unbalanced,name):
    fig, axs = plt.subplots(2,3,figsize=(10, 6))  # set size
    axs = axs.flatten()
    #shuffle colors list
    color_dict = make_color_dict(colors, dendrogram_balanced.partition_lc, dendrogram_balanced.partition_mcmc)
    
    balanced_lc_cut = 1-dendrogram_balanced.list_D[np.argmax([x[0] for x in dendrogram_balanced.list_D])][1]
    dendrogram_plot(axs[0],dendrogram_balanced.linkage,dendrogram_balanced.partition_lc,color_dict,lim_color=balanced_lc_cut,vertical_line=True,D=dendrogram_balanced.D_lc_max,title='LinkClustering dendrogram')
    dendrogram_plot(axs[1],dendrogram_balanced.linkage,dendrogram_balanced.partition_mcmc,color_dict,D=dendrogram_balanced.D_mcmc_max,title='AdaptiveCut dendrogram')
    plot_entropy(axs[2],dendrogram_balanced.real_entropy,dendrogram_balanced.max_entropy,dendrogram_balanced.min_entropy,dendrogram_balanced.entropy_levels,balanceness=dendrogram_balanced.balanceness)
    
    color_dict = make_color_dict(colors, dendrogam_unbalanced.partition_lc, dendrogam_unbalanced.partition_mcmc)
    
    unbalanced_lc_cut = 1-dendrogam_unbalanced.list_D[np.argmax([x[0] for x in dendrogam_unbalanced.list_D])][1]
    dendrogram_plot(axs[3],dendrogam_unbalanced.linkage,dendrogam_unbalanced.partition_lc,color_dict,lim_color=unbalanced_lc_cut,vertical_line=True,D=dendrogam_unbalanced.D_lc_max,title='Link Clustering Dendrogram')
    dendrogram_plot(axs[4],dendrogam_unbalanced.linkage,dendrogam_unbalanced.partition_mcmc,color_dict,D=dendrogam_unbalanced.D_mcmc_max,title='Adaptive Cut Dendrogram')
    plot_entropy(axs[5],dendrogam_unbalanced.real_entropy,dendrogam_unbalanced.max_entropy,dendrogam_unbalanced.min_entropy,dendrogam_unbalanced.entropy_levels,balanceness=dendrogam_unbalanced.balanceness)
    
    plt.tight_layout()
    plt.savefig(f'figures/compare_{name}__.png',dpi=300)
    
    
if __name__=='__main__':
    plot_compare_methods(dendrogram,dendrogram,'test_')
    
    
