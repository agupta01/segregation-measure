import numpy as np
import pandas as pd
import time
from shapely.geometry import Point
import matplotlib.pyplot as plt
%matplotlib inline
from shapely import speedups
speedups.enable()

tract_coords = pd.DataFrame({'tract_ID': ['A', 'B', 'C', 'D'],
                            'point': [Point(1, 3), Point(2,1), Point(4, 3), Point(3, 5)]})
chain = []
lines = []
tract_coords

points = tract_coords.point.values
distance_matrix = [[a.distance(b) for b in points] for a in points]
distance_matrix
np.random.seed(90)
len(points)
n0 = np.random.randint(len(points))
n0
prob_of_advance = lambda a, b: (1 - (distance_matrix[a][b] / sum(distance_matrix[a])))/2 if a != b else 0  # a is origin, b is destination
prob = [prob_of_advance(0, i) for i in range(len(points))]
prob
choices = np.random.choice([0,1,2,3], size = 10000, p=prob)
from scipy.stats import mode
plt.hist(choices)
