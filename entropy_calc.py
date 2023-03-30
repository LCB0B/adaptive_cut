import math

def entropy_calc(newcid2cids, num_edges, level):

    total_leaves = num_edges
    E, max_E = {}, {}

    E[0] = math.log(num_edges, 2)
    max_E[0] = math.log(num_edges, 2)

    for i, curr_partitions in enumerate(level.values()):

        logs = 0

        for leader in curr_partitions:

            # Find leaves
            if leader < num_edges:
                num_leaves = 1
            else:
                value = newcid2cids[leader]
                value = list(value)
                result = value.copy()
                for item in result:
                    if newcid2cids.get(item):
                        result.extend(newcid2cids[item])

                num_leaves = len(set([val for val in result if val < num_edges]))
                
            probj = num_leaves/total_leaves
            logs += -(probj * math.log(probj, 2))

        E[i] = logs
        max_E[i] = math.log(len(curr_partitions), 2)

    div = [v/max_E[k] for k, v in E.items() if k != (len(E)-1)]
    avg_div = sum(div)/len(div)

    sub = [max_E[k]-v for k, v in E.items() if k != (len(E)-1)]
    avg_sub = sum(sub)/len(sub)

    return E, max_E, div, avg_div, sub, avg_sub 

def balanceness(newcid2cids, num_edges, level):
    total_leaves = num_edges
    E, max_E = {}, {}
    E[0] = math.log(num_edges, 2)
    max_E[0] = math.log(num_edges, 2)
    for i, curr_partitions in enumerate(level.values()):
        logs = 0
        for leader in curr_partitions:
            # Find leaves
            if leader < num_edges:
                num_leaves = 1
            else:
                value = newcid2cids[leader]
                value = list(value)
                result = value.copy()
                for item in result:
                    if newcid2cids.get(item):
                        result.extend(newcid2cids[item])
                num_leaves = len(set([val for val in result if val < num_edges]))
            probj = num_leaves/total_leaves
            logs += -(probj * math.log(probj, 2))
        E[i] = logs
        max_E[i] = math.log(len(curr_partitions), 2)

    div = [v/max_E[k] for k, v in E.items() if k != (len(E)-1)]
    avg_div = sum(div)/len(div)
    
    sub = [max_E[k]-v for k, v in E.items() if k != (len(E)-1)]
    avg_sub = sum(sub)/len(sub)
    return E, max_E, div, avg_div, sub, avg_sub 