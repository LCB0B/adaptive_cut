from helper_functions import *
from logger import logger
import copy

def greedy_up(num_edges, groups, newcid2cids, cid2numedges, cid2numnodes):

    M = 2/num_edges
    groups_r = copy.deepcopy(groups)
    groups_r = dict(reversed(list(groups_r.items())))
    last_group = list(groups_r.values())[0]
    best_D = [0.0]
    partition_list, best_partitions, removed_comm = [], [], []

    for b_cid, group in groups_r.items():

        Dc_list = []

        # Do not look on the communities 
        # which had partition density less than the best
        if len(set(group).intersection(removed_comm)) > 0:

            removed_comm = removed_comm + [cid for cid in group if cid < num_edges] + [i for cid in group if cid >= num_edges for (key, value) in newcid2cids.items() for i in value if key == cid]
            
            add_removed_comm = [val for val in group for group in groups_r.values() if set(group) & set(removed_comm)]
            removed_comm = list(set(removed_comm + add_removed_comm))
            
            continue

        belonging_cid_lst = [key for g in group for (key, value) in newcid2cids.items() if g in value]
        if len(set(belonging_cid_lst).intersection(removed_comm)) > 0:

            removed_comm = removed_comm + [cid for cid in group if cid < num_edges] + [i for cid in group if cid >= num_edges for (key, value) in newcid2cids.items() for i in value if key == cid]
            
            add_removed_comm = [val for val in group for group in groups_r.values() if set(group) & set(removed_comm)]
            removed_comm = list(set(removed_comm + add_removed_comm))
            
            continue

        if group == last_group:
            for cid in group:
                m, n = cid2numedges[cid], cid2numnodes[cid]
                Dc_list.append(Dc(m, n))

            partition_list.append(set(group))

            D = M * sum(Dc_list)
        else:
            latest_partition_list = partition_list[-1]

            latest_partition_list = [c for c in latest_partition_list if c != b_cid]
            
            current_group = latest_partition_list + group     

            for cid in current_group:
                m, n = cid2numedges[cid], cid2numnodes[cid]
                Dc_list.append(Dc(m, n))
            
            partition_list.append(set(current_group))

            D = M * sum(Dc_list)

        if D < best_D[-1]:
            partition_list = partition_list[:-1]

            removed_comm = removed_comm + [i for cid in group for (key, value) in newcid2cids.items() for i in value if key == cid]

            add_removed_comm = [val for val in group for group in groups_r.values() if set(group) & set(removed_comm)]
            removed_comm = list(set(removed_comm + add_removed_comm))

        else:
            best_D.append(D)
            best_partitions = partition_list[-1]


    return best_D[-1], best_partitions


def greedy_bottom(num_edges, groups, orig_cid2edge, newcid2cids, cid2numedges, cid2numnodes):

    M = 2/num_edges
    best_D = [0.0]
    partition_list, best_partitions, removed_comm = [], [], []
    current_group = list(orig_cid2edge.keys())

    for b_cid, group in groups.items():

        Dc_list = []

        if len(set(group).intersection(removed_comm)) > 0:
            belonging_cid_lst = set([key for (key, value) in newcid2cids.items() if len(set(group).intersection(value)) > 0])

            removed_comm = list(removed_comm) + group + list(belonging_cid_lst) + [key for (key, value) in newcid2cids.items() if len(belonging_cid_lst.intersection(value)) > 0]
            removed_comm = set(removed_comm)

            continue

        current_group = current_group + [b_cid]
        current_group = [cid for cid in current_group if cid not in group]

        for cid in current_group:
            m, n = cid2numedges[cid], cid2numnodes[cid]
            Dc_list.append(Dc(m, n))
        
        partition_list.append(set(current_group))

        D = M * sum(Dc_list)

        if D < best_D[-1]:
            partition_list = partition_list[:-1]
            current_group = list(partition_list[-1])

            belonging_cid_lst = set([key for (key, value) in newcid2cids.items() if len(set(group).intersection(value)) > 0])

            removed_comm = list(removed_comm) + list(belonging_cid_lst) + [key for (key, value) in newcid2cids.items() if len(belonging_cid_lst.intersection(value)) > 0]
            removed_comm = set(removed_comm)

        else:
            best_D.append(D)
            best_partitions = partition_list[-1]

    return best_D[-1], best_partitions