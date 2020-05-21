import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import math
from sklearn.cluster import KMeans
from shapely.geometry import Point, LineString, MultiLineString

class TreeNode:
    def __init__(self, tracts=None):
        self.tracts = tracts
        self.left = None
        self.right = None
        self.parent = None
        self.heads = []
        # stores lines in subgraph once we begin merging
        self.graph = []
        # line that connects subgraph to another subgraph
        self.connecting_line = None

class ClusterTree:
    def __init__(self, all_tracts=None):
        self.root = TreeNode(all_tracts)
        self.size = 0

    def add(self, tracts):
        """
        Adds new node to tree. Sorting into tree is done by subset
        i.e. root is 'ABCDE', right is 'DE', left is 'ABC',
        then a new node consisting of 'BC' would be placed as a right
        child of 'ABC', since it is a subset of that list and is > 'ABCDE'

        Parameters:
        tracts - List of tract_IDs to include in this node
        """
        # create new Node to house tracts
        newNode = TreeNode(tracts)
        # helper function to check if new list is subset of parent one
        isSubset = lambda sub, main: all(x in main for x in sub)
        # make sure all tracts are part of main set
        assert(isSubset(tracts, self.root.tracts))

        currNode = self.root
        while (currNode != None):
            # assert(isSubset(tracts, currNode.tracts), "Node insertion failed: \
            # tracts aren't subset of parent node tracts")
            if (currNode.left == None and currNode.right == None):
                if (tracts < currNode.tracts):
                    currNode.left = newNode
                    newNode.parent = currNode
                    currNode = None
                elif (tracts > currNode.tracts):
                    currNode.right = newNode
                    newNode.parent = currNode
                    currNode = None
            elif (currNode.left == None):
                currNode.left = newNode
                newNode.parent = currNode
                currNode = None
            elif (currNode.right == None):
                currNode.right = newNode
                newNode.parent = currNode
                currNode = None
            else:
                if (isSubset(tracts, currNode.left.tracts)):
                    currNode = currNode.left
                else:
                    currNode = currNode.right

    # A function to do postorder tree traversal
    def printPostorder(self, root):
        if root:
            print(root.tracts)
            self.printPostorder(root.left)
            self.printPostorder(root.right)


def assemble_tree():
    global tract_coords
    # create tree with root being list of all tract_IDs
    tree = ClusterTree(list(tract_coords.tract_ID.values))

    # cluster through recursively and divide into two until each cluster has 1-2 tracts
    clustering(tree, tree.root.tracts)

    return tree


def merge(child_node_a, child_node_b):
    """
    Merges two nodes together. Assume subgraphs are merged.
    Note that the nature of the graph generation ensures that there are no single-child sub-graphs
    (all parents have 2 children).
    Args:
        child_node_a: first child node to merge
        child_node_b: second child node to merge, sibling to child_node_a
    """
    # look at the heads of the two graphs, make the pairs that can be merged
    # let a be a head from child a, b be a nofe from child b
    merge_pairs = [(a, b) for a in child_node_a.heads for b in child_node_b.heads]

    # evaluate pairs to see if they are "visible" to each other - i.e. if they
    # don't cross over either node's existing lines
    for p in merge_pairs:
        # TODO: check all lines against line made by p
        # TODO: if it intersects, remove from merge_pairs
        pass

    if (len(merge_pairs) == 0): # no "visible" connections, have to go back and redo merge children for both nodes
        # first, need to remove old lines from children's left & right
        for child in [child_node_a, child_node_b]:
            child.left.graph.remove(child.left.connecting_line)
            child.left.connecting_line = None
            child.right.graph.remove(child.right.connecting_line)
            child.right.connecting_line = None

        merge(child_node_a.left, child_node_a.right)
        merge(child_node_b.left, child_node_b.right)
    else:
        # copy lines and heads into parent, plus new line (randomly chosen from merge_pairs)
        choice = merge_pairs[np.random.choice(range(len(merge_pairs)))]
        connection = Line(choice[0], choice[1])
        child_node_a.connection_line = connection
        child_node_b.connection_line = connection
        child_node_a.parent.graph = child_node_a.graph + child_node_b.graph
        child_node_a.parent.graph.append(connection)
        # heads for parent are the 2 nodes that weren't connected
        child_node_a.parent.heads = [child_node_a.heads[1-choice[0]], child_node_b.heads[1-choice[1]]]
    # return parent as TreeNode object
    return child_node_a.parent # or child_node_b.parent, it really doesn't matter

def merge_tree(self, root):
    """
    Main function for merge, runs it recursively.
    Args:
        root - root node of subtree
    Returns:
        TreeNode object, with lines and head/tail listed out inside
        None, if graph is blank/only has root node
    """
    if root.left and root.right:
        merge(merge_tree(root.left), merge_tree(root.right))
    else:
        assert("Tree empty! No nodes to merge.")
        return None

def clustering(tree, tracts):
    global tract_coords
    if (len(tracts) < 3):
        # TODO
        # if 2 tracts, create line between them
        return 0
    else:
        # split into 2 groups using clustering
        tracts_to_split = tract_coords.loc[tract_coords.tract_ID.isin(tracts)]
        clusters = KMeans(n_clusters=2).fit_predict(np.column_stack([tracts_to_split.X.values, tracts_to_split.Y.values]))
        tracts_to_split['cluster'] = clusters
        group_a = tracts_to_split[tracts_to_split.cluster == 0]
        group_b = tracts_to_split[tracts_to_split.cluster == 1]

        # create 2 new nodes and add to tree, with groups
        tree.add(list(group_a.tract_ID.values))
        tree.add(list(group_b.tract_ID.values))

        # cluster both new groups
        clustering(tree, list(group_a.tract_ID.values))
        clustering(tree, list(group_b.tract_ID.values))


tract_coords = pd.DataFrame({'tract_ID': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                          'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'],
                             'X': [1, 2, 4, 3, 5, 10, 0, 9, 0, 0, 7, 5, 2, 7, 0, 9],
                             'Y': [3, 1, 3, 5, 6, 11, 9, 0, 1, 0, 3, 10, 8, 8, 10, 5]})


def main():
    cluster_tree = assemble_tree()
    cluster_tree.printPostorder(cluster_tree.root)

if __name__ == '__main__':
    main()
