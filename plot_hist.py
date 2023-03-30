import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')

plt.rcParams['font.size'] = 10

# also need all of the children

#then plot distirbutions

path = '/Users/louibo/phd/adaptive_cut/output/imgs/'
dataset_name = 'Epinions'
#dataset_name = 'Pretty Good Privacy'

#dataset_name = 'Political books'
partition_mc = np.load(path+f'{dataset_name}/labels_mc.npy',allow_pickle='TRUE')
partition_lc = np.load(path+f'{dataset_name}/labels_lc.npy',allow_pickle='TRUE')



lc = [ v for k,v in (partition_lc.item()).items()]
mc = [ v for k,v in (partition_mc.item()).items()]




lc_c = np.unique(lc, return_counts=True)
mc_c = np.unique(mc, return_counts=True)

fig, axs = plt.subplots(1, 1,figsize=(4,3),sharey=True, sharex='col')
# axs[0].hist(lc_c[1],bins=np.arange(mc_c[1].max()))
# axs[1].hist(mc_c[1],bins=np.arange(mc_c[1].max()))
bins = np.logspace(0,np.log(mc_c[1].max())/np.log(10),60)

lc_b, bins_b = np.histogram(lc_c[1],bins=bins)
mc_b, _ = np.histogram(mc_c[1],bins=bins)

axs.plot(bins_b[:-1],lc_b,'^',color='dodgerblue',markersize=3,label='Link Clustering',alpha=0.6)
axs.plot(bins_b[:-1],mc_b,'8',color='firebrick',markersize=3,label='Adaptive Cut',alpha=0.6)
axs.legend()
axs.set_xlabel('Cluster size')
axs.set_ylabel(r'Number of cluster')
# axs[1].set_xlabel('Balance Ratio 0-1')
# axs[1].set_ylabel(r'Balance Subtraction 0-$\inf$')
for j in range(1):
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_yscale('log')
    axs.set_xscale('log')
plt.tight_layout()
#plt.show()
plt.savefig(path+dataset_name+'/cluster_dist_comp.png')

