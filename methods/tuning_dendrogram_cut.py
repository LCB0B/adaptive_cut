from helper_functions import *
import random
from copy import deepcopy
from logger import logger 

def partition_density(num_edges, cid2numedges, cid2numnodes, partitions):
    '''
    Calculate the Partition Density 

    Input:
        - partitions: the partitions for which we want to calculate the partition density
    Output:
        - D: partition density value
    '''

    M = 2/num_edges

    Dc_list = []
    for cid in partitions:
        m, n = cid2numedges[cid], cid2numnodes[cid]
        Dc_list.append(Dc(m, n))

    D = M * sum(Dc_list)

    return D

def partition_density_dist(num_edges, cid2numedges, cid2numnodes, partitions):
    M = 2/num_edges
    D =[]
    Dc_list = []
    for cid in partitions:
        m, n = cid2numedges[cid], cid2numnodes[cid]
        Dc_list.append([Dc2(m, n),m])
    return Dc_list


def calc_partdens_up(curr_leader, num_edges, groups, cid2numedges, cid2numnodes, newcid2cids, curr_partitions):
    '''
    Identify the partitions in case we have to move the current leader one level up. 
    This means that we have to identify the community id in which the current leader
    belongs to and remove from the current partitions any children belonging to this 
    community id.

    Based on the extracted partitions, we calculate the Partition Density.

    Input:
        - curr_leader: the id of the current leader
        - newcid2cids: dictionary which shows from which pair of cids each new cid has been created
        - curr_partitions: the list of leader ids from the current partitions 
    Output:
        - partitions: list of the identified partitions' leaders
        - D: calculated partition density
    '''

    # find belonging cid
    belonging_cid = [k for k, v in groups.items() if curr_leader in v][0]

    # find belonging cid's children
    value = newcid2cids[belonging_cid]
    value = list(value)
    result = value.copy()
    for item in result:
        if newcid2cids.get(item):
            result.extend(newcid2cids[item])

    # add belonging cid in partitions 
    # remove belonging cid's children from partitions
    partitions = [leader_tmp for leader_tmp in curr_partitions if leader_tmp not in result]
    partitions = [leader_tmp for leader_tmp in partitions if leader_tmp not in groups[belonging_cid]]
    partitions.append(belonging_cid)

    # Calculate partition density
    D = partition_density(num_edges, cid2numedges, cid2numnodes, partitions)

    return partitions, D


def calc_partdens_down(curr_leader, num_edges, cid2numedges, cid2numnodes, groups, curr_partitions):
    '''
    Identify the partitions in case we have to move the current leader one level down. 
    This means that we have to identify the children ids which the current leader
    has and remove from the current partitions the current leader.

    Based on the extracted partitions, we calculate the Partition Density.

    Input:
        - curr_leader: the id of the current leader
        - newcid2cids: dictionary which shows from which pair of cids each new cid has been created
        - curr_partitions: the list of leader ids from the current partitions 
    Output:
        - partitions: list of the identified partitions' leaders
        - D: calculated partition density
    '''

    # remove current leader
    partitions = [leader_tmp for leader_tmp in curr_partitions if leader_tmp != curr_leader]

    # find children
    group_down = groups[curr_leader]

    # update partitions
    partitions = partitions + group_down

    # Calculate partition density
    D = partition_density(num_edges, cid2numedges, cid2numnodes, partitions)

    return partitions, D


def tune_cut(similarity_value, best_D, cid2numedges, cid2numnodes, newcid2cids, groups, num_edges, level, threshold, stopping_threshold=None, montecarlo=False, epsilon=None):

    leaders = level[similarity_value] 

    direction = ['up', 'down']
    curr_partitions = deepcopy(leaders)
    early_stop = 0

    i = 0

    list_D, list_clusters = {}, {}
    list_D[i] = best_D
    list_clusters[i] = len(curr_partitions)

    only_down_lst, only_up_lst = [], []

    while True:
        
        curr_leader = random.choice(curr_partitions)

        # find belonging cid
        belonging_cid = [k for k, v in groups.items() if curr_leader in v][0]

        if belonging_cid == max(list(newcid2cids.keys())) or belonging_cid in list(groups.values())[-1]:
            curr_direction = 'down'

        elif (len(only_down_lst)>0 or len(only_up_lst)>0):
            if curr_leader in only_down_lst:
                curr_direction = 'down'
            elif curr_leader in only_up_lst:
                curr_direction = 'up'
            else:
                curr_direction = random.choice(direction)
        else:
            curr_direction = random.choice(direction)

        i += 1
        if curr_direction == 'up':
            # Move one level up   
            partitions_tmp, curr_D = calc_partdens_up(curr_leader, num_edges, groups, cid2numedges, cid2numnodes, newcid2cids, curr_partitions)
        else:
            # Move one level down
            if curr_leader < num_edges:
                partitions_tmp, curr_D = curr_partitions, best_D
            else:
                partitions_tmp, curr_D = calc_partdens_down(curr_leader, num_edges, cid2numedges, cid2numnodes, groups, curr_partitions)

        previous_D = best_D

        if curr_D >= best_D:
            if curr_direction == 'up':
                only_down_lst = list(set(only_down_lst + list(set(partitions_tmp).difference(curr_partitions))))
            else:
                only_up_lst = list(set(only_up_lst + list(set(partitions_tmp).difference(curr_partitions))))
            curr_partitions = partitions_tmp
            best_D = curr_D
            list_D[i] = best_D
            list_clusters[i] = len(curr_partitions)
        else:
            if montecarlo:
                a = random.uniform(0, 1)

                if (a < epsilon): 
                    if curr_direction == 'up':
                        only_down_lst = list(set(only_down_lst + list(set(partitions_tmp).difference(curr_partitions))))
                    else:
                        only_up_lst = list(set(only_up_lst + list(set(partitions_tmp).difference(curr_partitions))))
                    curr_partitions = partitions_tmp
                    best_D = curr_D
                    list_D[i] = best_D
                    list_clusters[i] = len(curr_partitions)

                if i > threshold:
                    break

        if not montecarlo:
            if (i > threshold) and ((best_D - previous_D) < 0.01):
                early_stop += 1
            else:
                early_stop = 0

            if early_stop > stopping_threshold:
                break

    return list_D, list_clusters, curr_partitions





# def mc_tune_cut(similarity_value, best_D, cid2numedges, cid2numnodes, newcid2cids, groups, num_edges, level, threshold, stopping_threshold=None, montecarlo=False, epsilon=None):
#     leaders = level[similarity_value] 
#     direction = ['up', 'down']
#     curr_partitions = deepcopy(leaders)
#     early_stop = 0
#     i = 0
#     T0 = 2
#     list_D, list_clusters = {}, {}
#     list_D[i] = best_D
#     r_best_D = best_D
#     list_clusters[i] = len(curr_partitions)
#     only_down_lst, only_up_lst = [], []
#     best_partitions = curr_partitions
#     C = 1/np.log(num_edges)
#     while True:
#         curr_leader = random.choice(curr_partitions) #choose point that go up/down
#         # find belonging cid
#         belonging_cid = [k for k, v in groups.items() if curr_leader in v][0]
#         if belonging_cid == max(list(newcid2cids.keys())) or belonging_cid in list(groups.values())[-1]:
#             curr_direction = 'down'
#         elif (len(only_down_lst)>0 or len(only_up_lst)>0):
#             if curr_leader in only_down_lst:
#                 curr_direction = 'down'
#             elif curr_leader in only_up_lst:
#                 curr_direction = 'up'
#             else:
#                 curr_direction = random.choice(direction)
#         else:
#             curr_direction = random.choice(direction)
#         i += 1
#         if curr_direction == 'up':
#             # Move one level up   
#             partitions_tmp, curr_D = calc_partdens_up(curr_leader, num_edges, groups, cid2numedges, cid2numnodes, newcid2cids, curr_partitions)
#         else:
#             # Move one level down
#             if curr_leader < num_edges:
#                 partitions_tmp, curr_D = curr_partitions, best_D
#             else:
#                 partitions_tmp, curr_D = calc_partdens_down(curr_leader, num_edges, cid2numedges, cid2numnodes, groups, curr_partitions)

#         if curr_D >r_best_D:
#             best_partitions = partitions_tmp
#         # simulated annealing update 
        
#         T = 1 / (C*np.log(T0+i))
#         # markov chain ratio
#         ratio = (np.exp(curr_D/T)/ np.exp(best_D/T)) *(2*(curr_direction=='up')+(1/2)*(curr_direction=='down'))
#         alpha = min(1,ratio)
#         #print(alpha,curr_D,best_D)
#         if np.random.uniform()<alpha:
#             curr_partitions = partitions_tmp
#             best_D = curr_D
#             list_D[i] = best_D
#             list_clusters[i] = len(curr_partitions)
#         else :
#             list_D[i] = best_D
#             list_clusters[i] = len(curr_partitions)

#         if i > threshold:
#             break
#         r_best_D = max(list(list_D.values()))

#     best_index = [k for k, v in list_D.items() if v == r_best_D][-2]
#     return list_D, list_clusters, best_partitions,best_index 


        

def mc_tune_cut_2(similarity_value, best_D, cid2numedges, cid2numnodes, newcid2cids, groups, num_edges, level, threshold, stopping_threshold=None, montecarlo=False, epsilon=None):
    leaders = level[similarity_value] 
    direction = ['up', 'down']
    curr_partitions = deepcopy(leaders)
    early_stop = 0
    i = 0
    list_D, list_clusters = {}, {}
    list_D[i] = best_D
    curr_D = best_D
    list_clusters[i] = len(curr_partitions)
    only_down_lst, only_up_lst = [], []
    best_partitions = curr_partitions
    best_i=0
    ratio_list = [0]*int(threshold)
    #T0 = 10**4
    #C= 200
    i_lim = 1
    (T0,C)=epsilon
    direction = False
    while True:
        i += 1
        if i > int(threshold)-2:
            break

        #prev_D = curr_D
        curr_leader = random.choice(curr_partitions) #choose point that go up/down
        try :
            belonging_cid = [k for k, v in groups.items() if curr_leader in v][0]
        except:
            curr_direction = 'down'
            direction = True

        if curr_leader < num_edges:
            curr_direction = 'up'
            direction = True

        #if we are on the last level, we can only go down
        if not direction:
            if belonging_cid == max(list(newcid2cids.keys())) or belonging_cid in list(groups.values())[-1]:
                curr_direction = 'down'
                #print(f'ohoh I am on top and i={i}')
            elif i> min(threshold//2,1): # only up threshold 
                curr_direction = random.choice(direction)
            else :
                curr_direction= 'up'



        if curr_direction == 'up':
            # Move one level up   
            partitions_tmp, tmp_D = calc_partdens_up(curr_leader, num_edges, groups, cid2numedges, cid2numnodes, newcid2cids, curr_partitions)
        else:
            partitions_tmp, tmp_D = calc_partdens_down(curr_leader, num_edges, cid2numedges, cid2numnodes, groups, curr_partitions)

        direction = False

        if i<i_lim :
            T = 1 / (C*(np.log(T0)))
        else:
            T = 1 / (C*(np.log(T0+i)))
        # markov chain ratio
        try:
            ratio = (np.exp((tmp_D-curr_D)/T))*((2*len(partitions_tmp)/len(curr_partitions))**(2*(curr_direction=='down')-1))
            #ratio = (np.exp(curr_D/T)/ np.exp(best_D/T)) *(1*(curr_direction=='up')+(1/2)*(curr_direction=='down'))
        except:
            ratio = -0.1
        alpha = min(1,ratio)
        ratio_list[i] = alpha
        #print(alpha,curr_D,best_D)
        if np.random.uniform()<alpha:
            curr_D = tmp_D
            curr_partitions = partitions_tmp          
            list_D[i] = curr_D
            list_clusters[i] = len(curr_partitions)
            if tmp_D >best_D:
                best_partitions = partitions_tmp
                print(f'update best {i} et {tmp_D,best_D}')
                best_D=tmp_D
                best_i=i
            #elif curr_D > tmp_D:
                #print(f'not update {i}, alpha {alpha} ')
        else :
            list_D[i] = curr_D
            list_clusters[i] = len(curr_partitions)


        #r_best_D = max(list(list_D.values()))

    #print(f'best _i: {best_i}')
    #best_index = [k for k, v in list_D.items() if v == best_D][-1]
    return list_D, list_clusters, best_partitions,best_i,ratio_list,curr_partitions





def mc_tune_cut_all(similarity_value, best_D, cid2numedges, cid2numnodes, newcid2cids, groups, num_edges, level, threshold, stopping_threshold=None, montecarlo=False, epsilon=None):
    leaders = level[similarity_value] 
    direction = ['up', 'down']
    curr_partitions = deepcopy(leaders)
    early_stop = 0
    i = 0
    list_D, list_clusters = {}, {}
    list_D[i] = best_D
    curr_D = best_D
    list_clusters[i] = len(curr_partitions)
    only_down_lst, only_up_lst = [], []
    best_partitions = curr_partitions
    best_i=0
    ratio_list = [0]*int(threshold)
    #T0 = 10**4
    #C= 200
    i_lim = 10**4
    (T0,C)=epsilon
    dir = False
    while True:
        i += 1
        if i > int(threshold)-2:
            break

        #prev_D = curr_D
        curr_leader = random.choice(curr_partitions) #choose point that go up/down
        if curr_leader < num_edges:
            curr_direction = 'up'
            dir = True
        elif i> min(threshold//2,1): # only up threshold 
            curr_direction = random.choice(direction)
        else :
            curr_direction= 'up'

        if dir == False:
            try :
                belonging_cid = [k for k, v in groups.items() if curr_leader in v][0]
                if belonging_cid == max(list(newcid2cids.keys())) or belonging_cid in list(groups.values())[-1]:
                    curr_direction = 'down'
            except:
                a=1
                #print(curr_leader)
        dir = False

        if curr_direction == 'up':
            # Move one level up   
            try:
                partitions_tmp, tmp_D = calc_partdens_up(curr_leader, num_edges, groups, cid2numedges, cid2numnodes, newcid2cids, curr_partitions)
            except Exception as e:
                partitions_tmp, tmp_D = calc_partdens_down(curr_leader, num_edges, cid2numedges, cid2numnodes, groups, curr_partitions)
        else:
            partitions_tmp, tmp_D = calc_partdens_down(curr_leader, num_edges, cid2numedges, cid2numnodes, groups, curr_partitions)

        if i<i_lim :
            T = 1 / (C*(np.log(T0)))
        else:
            T = 1 / (C*(np.log(T0+i-i_lim)))
        # markov chain ratio
        try:
            ratio = (np.exp((tmp_D-curr_D)/T))*((2*len(partitions_tmp)/len(curr_partitions))**(2*(curr_direction=='down')-1))
            ratio = (np.exp((tmp_D-curr_D)/T))*((2)**(2*(curr_direction=='down')-1))
            #ratio = (np.exp(curr_D/T)/ np.exp(best_D/T)) *(1*(curr_direction=='up')+(1/2)*(curr_direction=='down'))
        except:
            ratio = -.1
        alpha = min(1,ratio)
        ratio_list[i] = alpha
        #print(alpha,curr_D,best_D)
        if np.random.uniform()<alpha:
            curr_D = tmp_D
            curr_partitions = partitions_tmp          
            list_D[i] = curr_D
            list_clusters[i] = len(curr_partitions)
            if tmp_D >best_D:
                best_partitions = partitions_tmp
                print(f'update best {i} et {tmp_D,best_D}')
                best_D=tmp_D
                best_i=i
            #elif curr_D > tmp_D:
                #print(f'not update {i}, alpha {alpha} ')
        else :
            list_D[i] = curr_D
            list_clusters[i] = len(curr_partitions)

        if i > int(threshold)-2:
            break

        #r_best_D = max(list(list_D.values()))

    #print(f'best _i: {best_i}')
    #best_index = [k for k, v in list_D.items() if v == best_D][-1]
    return list_D, list_clusters, best_partitions,best_i,ratio_list,curr_partitions

