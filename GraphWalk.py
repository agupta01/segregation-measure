import numpy as np
import pandas as pd
import time
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
%matplotlib inline
from shapely import speedups
speedups.enable()

# dataframe containing points to traverse
tract_coords = pd.DataFrame({'tract_ID': ['A', 'B', 'C', 'D'],
                            'point': [Point(1, 3), Point(2,1), Point(4, 3), Point(3, 5)]})
# points in order of traversal
chain = []
# lines formed by traversals
lines = []
tract_coords

points = tract_coords.point.values
# compute distance matrix (from a to b)
distance_matrix = [[a.distance(b) for b in points] for a in points]
distance_matrix
# start at random point
np.random.seed(90)
len(points)
n0 = np.random.randint(len(points))
n0
# generate traversal probabilities
prob_of_advance = lambda a, b: (1 - (distance_matrix[a][b] / sum(distance_matrix[a])))/2 if a != b else 0  # a is origin, b is destination
prob = [prob_of_advance(n0, i) for i in range(len(points))]
prob
choices = np.random.choice([0,1,2,3], size = 10000, p=prob)
from scipy.stats import mode
n1 = max(choices)

# add new choice to chain
chain.append(n0)
chain.append(n1)

# add new line to list of lines
n0_coords = list(tract_coords.iloc[n0].point.coords)[0]
n1_coords = list(tract_coords.iloc[n1].point.coords)[0]
n1_coords
lines.clear()
line_is_unique(n0, n1)
lines

# check if given line crosses between existing line on chain
line1 = LineString([(0, 0), (1, 1)])
line2 = LineString([(1, 0), (0, 1)])
print(line1.intersects(line2))

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
    for line in lines:
        if line.intersects(line_to_check):
            return False
    return True
