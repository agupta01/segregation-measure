import numpy as np
import time
import pandas as pd
from scipy.stats import mode
from shapely.geometry import Point, LineString, MultiLineString
import matplotlib.pyplot as plt
# %matplotlib inline
from shapely import speedups
speedups.enable()

MAX_LIMIT = 1e32
np.random.seed()

# dataframe containing points to traverse
tract_coords = pd.DataFrame({'tract_ID': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                            'point': [Point(1, 3), Point(2,1), Point(4, 3), Point(3, 5), Point(5, 6), Point(10, 11), Point(0, 9), Point(9, 0)]})
points = tract_coords.point.values
index_to_ID = {tract_coords.index.values[i]:tract_coords.tract_ID.values[i] for i in tract_coords.index}
# tract_ID of points in order of traversal
chain = []
# lines formed by traversals
lines = []
# compute distance matrix (from a to b)
distance_matrix = [[a.distance(b) for b in points] for a in points]

def line_is_unique(a, b):
    """
    Checks if line between two nodes intersects any previous lines.
    
    Parameters:
        a (int): index of a node in DataFrame
        b (int): index of b node in DataFrame
    
    Returns:
        boolean: true if line doesn't intersect anything, false if it does
    """
    a_coords = list(tract_coords.iloc[a].point.coords)[0]
    b_coords = list(tract_coords.iloc[b].point.coords)[0]
    line_to_check = LineString([a_coords, b_coords])
    for line in lines[:-1]:
        if line.intersects(line_to_check):
            return False
    return True

def traverse(node_idx, blacklist=[], sample_size=10000):
    """
    Traverses from given node to next node of highest probability
    (probabilities weighted inverse to distance).
    
    Parameters:
        node_idx (int): index of node in DF to search from
        blacklist (int array): contains node indices which have been recorded as being inaccessible b/c behind a line
        sample_size (int): number of times to sample from probability distribution (smaller = more uncertainty)
    
    Returns:
        int: index of next node in DF
    """
    #print current chain
    print(chain)
    # filter distance_matrix to row of source node, and only include untraversed destination nodes
    # to filter, set traversed nodes to distance of MAX_LIMIT, so prob of moving to them next is so small it doesn't matter
    row = [distance_matrix[node_idx][i] if (index_to_ID[i] not in chain) and (i not in blacklist) else MAX_LIMIT for i in range(len(distance_matrix[node_idx]))]
    
    # compute intermediate terms of calculation
    max_d = max(row)
    phi = 1 / sum([max_d / d for d in row])
    
    # assemble probability vector
    prob_vector = np.array([(max_d/d)*phi for d in row])
    print("Probabilities: " + str(prob_vector))

    choices = np.random.choice(tract_coords.index.values, size = sample_size, p=prob_vector)
    
    # TODO: if all elements of choices are equal, node is "trapped". return -1 so we know.
    
    # if (len(set(choices)) == 1): 
    #     return -1
    return mode(choices)[0][0]

# traverse(n0)

chain.clear()
lines.clear()
node = np.random.randint(0, len(points) - 1)
# np.random.randint(len(points))
node
chain.append(index_to_ID[node])
while (len(chain) < len(points)):
    # traverse to next node
    print("Current node:", tract_coords.iloc[node].tract_ID)
    new_node = traverse(node)
    
    blacklist = []
    while (line_is_unique(node, new_node) == False):
        print(f"Line from {index_to_ID[node]} to {index_to_ID[new_node]} not unique!")
        blacklist.append(new_node)
        new_node = traverse(node, blacklist)
        if (len(blacklist) == len(points) - len(chain)):
            print("Finding failed! Reverting to last node")
            # # remove current node from chain, remove last line from lines, set current node as last node in chain
            # chain.remove(chain[-1])
            # node = chain[-1]
            # lines.remove(lines[-1])
            # blacklist.clear()
            break
    
    print("New node found!:", tract_coords.iloc[new_node].tract_ID)
    # add node and new line to data objects
    # chain.append(tract_coords.iloc[new_node].tract_ID)
    lines.append(LineString([list(tract_coords.iloc[node].point.coords)[0],
                            list(tract_coords.iloc[new_node].point.coords)[0]]))
    # add node to chain, since you know it has been traversed
    chain.append(index_to_ID[new_node])
    
    # set next node as current node
    node = new_node

print("\n\n\n" + str(chain))
print([line.xy for line in lines])

fig, ax = plt.subplots()
for i in range(len(points)):
    ax.scatter(points[i].x, points[i].y, label=tract_coords.tract_ID[i])
for line in lines:
    x, y = line.xy
    ax.plot(x, y, color='grey', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
ax.legend()
plt.show()
