import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import math
from sklearn.cluster import KMeans

class TreeNode:
    """Node object for cluster tree."""
    
    def __init__(self, parent, graph_nodes):
        """
        Creates new node.
        Parameters:
        parent - TreeNode directly above new node
        graph_nodes - list of graph nodes to add to this node
        
        """
        # self.children should contain no more than 2 TreeNode objects
        self.children = []
        self.parent = parent
        parent.children.append(self)
        self.graph_nodes = graph_nodes
        self.conn_to_sibling = False
        self.leaves_connected = False
    
    def update_status(self):
        """
        1. Change conn_to_sibling to true if all nodes in graph_nodes
        are connected to sibling at same level.
        2. Change leaves_connected if both child nodes are connected, or
        if children list is empty (node is a leaf)
        """
        assert(len(self.children) <= 2)
        
        if (self.is_leaf()):    
            self.leaves_connected = True
        if (self.children[0].conn_to_sibling and self.children[1].conn_to_sibling):
            self.leaves_connected = True
        
        self.get_sibling().graph_nodes
         
    def get_sibling(self):
        """Returns sibling of current node."""
        return [n for n in self.parent.children if n != self][0]
    
    def is_leaf(self):
        """Returns true if node is a leaf (no children)."""
        return not self.children
        
tract_coords = pd.DataFrame({'tract_ID': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'], 'X': [1, 2, 4, 3, 5, 10, 0, 9, 0, 0, 7, 5, 2, 7, 0, 9], 'Y': [3, 1, 3, 5, 6, 11, 9, 0, 1, 0, 3, 10, 8, 8, 10, 5]})

for i in range(int(math.log2(len(tract_coords.tract_ID)))):
    clusters = KMeans(n_clusters = 2*(i+1)).fit_predict(np.column_stack([tract_coords.X.values, tract_coords.Y.values]))
    tract_coords['cluster_{}'.format(i+1)] = clusters
    plt.scatter(tract_coords.X, tract_coords.Y, c=tract_coords['cluster_{}'.format(i+1)])
    plt.show()

tract_coords
