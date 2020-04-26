import numpy as np

import pandas as pd
from scipy.stats import mode
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
# %matplotlib inline
from shapely import speedups
speedups.enable()

MAX_LIMIT = 9999e10

# dataframe containing points to traverse
tract_coords = pd.DataFrame({'tract_ID': ['A', 'B', 'C', 'D'],
                            'point': [Point(1, 3), Point(2,1), Point(4, 3), Point(3, 5)]})
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

def traverse(node_idx, sample_size=1):
    """
    Traverses from given node to next node of highest probability
    (probabilities weighted inverse to distance).
    
    Parameters:
        node_idx (int): index of node in DF to search from
    
    Returns:
        int: index of next node in DF
    """
    # add node to chain, since you know it has been traversed
    chain.append(index_to_ID[node_idx])
    # filter distance_matrix to row of source node, and only include untraversed destination nodes
    # to filter, set traversed nodes to distance of MAX_LIMIT, so prob of moving to them next is so small it doesn't matter
    row = [distance_matrix[node_idx][i] if (index_to_ID[i] not in chain) else MAX_LIMIT for i in range(len(distance_matrix[node_idx]))]
    
    # compute intermediate terms of calculation
    max_d = max(row)
    phi = 1 / sum([max_d / d for d in row])
    
    # assemble probability vector
    prob_vector = np.array([(max_d/d)*phi for d in row])

    choices = np.random.choice(tract_coords.index.values, size = sample_size, p=prob_vector)
    return mode(choices)[0][0]

# traverse(n0)

chain.clear()
lines.clear()
node = 0
# np.random.randint(len(points))
node
chain.append(tract_coords.iloc[node].tract_ID)
while (len(chain) < len(points)):
    # traverse to next node
    print("Current node:", tract_coords.iloc[node].tract_ID)
    new_node = traverse(node)
    
    count = 0
    while (line_is_unique(node, new_node) == False):
        new_node = traverse(node)
        count = count + 1
    
    print("New node found!:", tract_coords.iloc[new_node].tract_ID)
    # add node and new line to data objects
    # chain.append(tract_coords.iloc[new_node].tract_ID)
    lines.append(LineString([list(tract_coords.iloc[node].point.coords)[0],
                            list(tract_coords.iloc[new_node].point.coords)[0]]))
    # set next node as current node
    print(chain, new_node)
    node = new_node
