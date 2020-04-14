import numpy as np
import pandas as pd
import time
from scipy.stats import mode
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
# %matplotlib inline
from shapely import speedups
speedups.enable()

# dataframe containing points to traverse
tract_coords = pd.DataFrame({'tract_ID': ['A', 'B', 'C', 'D'],
                            'point': [Point(1, 3), Point(2,1), Point(4, 3), Point(3, 5)]})
# points in order of traversal
chain = []
# lines formed by traversals
lines = []
# tract_coords
# 
points = tract_coords.point.values
# compute distance matrix (from a to b)
distance_matrix = [[a.distance(b) for b in points] for a in points]
# distance_matrix
# # start at random point
# np.random.seed(90)
# len(points)
# n0 = np.random.randint(len(points))
# n0
def line_is_unique(a, b):
    """
    checks if line between two nodes intersects any previous lines
    :params: a - index of a node in DataFrame
    b - index of b node in DataFrame
    :returns: boolean value, true if line doesn't intersect anything
    """
    a_coords = list(tract_coords.iloc[a].point.coords)[0]
    b_coords = list(tract_coords.iloc[b].point.coords)[0]
    line_to_check = LineString([a_coords, b_coords])
    for line in lines[:-1]:
        if line.intersects(line_to_check):
            return False
    return True

def traverse(node, sample_size=1):
    """
    traverses from given node to next node of highest probability
    (probabilities weighted inverse to distance)
    :params: node - index of node in DF to search from
    :returns: index of next node in DF
    """
    # filter DF to only nodes NOT in chain, plus given node
    origin_dist = distance_matrix[node]
    origin_dist = {i:origin_dist[i] for i in range(len(origin_dist)) 
    if (tract_coords.iloc[i].tract_ID not in chain) and (i != node)}
    phi = 1 / sum([max(list(origin_dist.values()))/d for d in origin_dist.values()])
    prob = np.zeros(len(distance_matrix[node]))
    for i in range(len(prob)):
        if i in origin_dist.keys():
            prob[i] = (max(list(origin_dist.values())) / origin_dist[i])*phi
    choices = np.random.choice(tract_coords.index.values, size = sample_size, p=prob)
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
    chain.append(tract_coords.iloc[new_node].tract_ID)
    lines.append(LineString([list(tract_coords.iloc[node].point.coords)[0],
                            list(tract_coords.iloc[new_node].point.coords)[0]]))
    # set next node as current node
    print(chain, new_node)
    node = new_node
