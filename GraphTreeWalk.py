import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import math
from sklearn.cluster import KMeans
from shapely import ops
from shapely.geometry import Point, LineString, MultiLineString
from tqdm import tqdm
from shapely import speedups

speedups.enable()


class TreeNode:
    def __init__(self, tracts=None, is_leaf=False):
        self.tracts = tracts
        self.left = None
        self.right = None
        self.parent = None
        self.heads = []
        # stores lines in subgraph once we begin merging
        self.graph = []
        # line that connects subgraph to another subgraph
        self.connecting_line = None
        self.is_leaf = is_leaf

    def graph_node(self, save=None):
        """
        Plots graph corresponding to node at time of invocation.
        Args:
            save: (optional) filepath to save graph image to, if desired

        Returns:
            None
        """
        global tract_coords
        fig, ax = plt.subplots()
        ax.scatter(tract_coords.X, tract_coords.Y, alpha=0.6)
        for i in range(len(self.tracts)):
            ax.text(tract_coords.X[i], tract_coords.Y[i], tract_coords.tract_ID[i])
        for line in self.graph:
            x, y = line.xy
            ax.plot(x, y, color='grey', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
        # ax.legend()
        if save != None:
            plt.savefig(save)
        plt.show()
        plt.close()


class ClusterTree:
    def __init__(self, all_tracts=None):
        self.root = TreeNode(all_tracts)

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
        newNode = TreeNode(tracts, is_leaf=True) if len(tracts) <= 2 else TreeNode(tracts)
        # add lines and heads if leaf node
        if len(tracts) == 2:
            newNode.heads = tracts
            a_coords = [tract_coords[tract_coords.tract_ID == tracts[0]].X.values[0],
                        tract_coords[tract_coords.tract_ID == tracts[0]].Y.values[0]]
            b_coords = [tract_coords[tract_coords.tract_ID == tracts[1]].X.values[0],
                        tract_coords[tract_coords.tract_ID == tracts[1]].Y.values[0]]
            newNode.graph.append(LineString([a_coords, b_coords]))
        elif len(tracts) == 1:
            newNode.heads = tracts
        # helper function to check if new list is subset of parent one
        isSubset = lambda sub, main: all(x in main for x in sub)
        # make sure all tracts are part of main set
        assert (isSubset(tracts, self.root.tracts))

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
            print(root.tracts, root.graph, root.heads)
            self.printPostorder(root.left)
            self.printPostorder(root.right)


def assemble_tree():
    global tract_coords
    # create tree with root being list of all tract_IDs
    tree = ClusterTree(list(tract_coords.tract_ID.values))

    # cluster through recursively and divide into two until each cluster has 1-2 tracts
    clustering(tree, tree.root.tracts)

    return tree


def merge(child_node_a, child_node_b, blacklist=[]):
    """
    Merges two nodes together. Assume subgraphs are merged.
    Note that the nature of the graph generation ensures that there are no single-child sub-graphs
    (all parents have 2 children).
    Args:
        child_node_a: first child node to merge
        child_node_b: second child node to merge, sibling to child_node_a
    """
    global tract_coords
    # look at the heads of the two graphs, make the pairs that can be merged
    # let a be a head from child a, b be a nofe from child b
    merge_pairs = [(a, b) for a in child_node_a.heads for b in child_node_b.heads]

    # evaluate pairs to see if they are "visible" to each other - i.e. if they
    # don't cross over either node's existing lines
    final_merge_pairs = []
    for p in merge_pairs:
        p0_coords = [tract_coords[tract_coords.tract_ID == p[0]].X.values[0],
                     tract_coords[tract_coords.tract_ID == p[0]].Y.values[0]]
        p1_coords = [tract_coords[tract_coords.tract_ID == p[1]].X.values[0],
                     tract_coords[tract_coords.tract_ID == p[1]].Y.values[0]]
        newLine = LineString([p0_coords, p1_coords])
        # construct ring using a.graph and b.graph and newLine, and see if it forms a ring
        # if so, remove it from list of candidates
        ring = MultiLineString(child_node_a.graph + child_node_b.graph + [newLine])
        if ring.is_simple:
            final_merge_pairs.append(p)

    merge_pairs = [p for p in final_merge_pairs]

    if (len(merge_pairs) == 0):  # no "visible" connections, have to go back and redo merge children for both nodes
        # first, need to remove old lines from children's left & right
        # raise RuntimeError("Trapped case reached!")
        print("Backtracking!")
        if len(child_node_a.tract) > 3:
            child_node_a.graph = []
            child_node_a.heads = []
            child_node_a.left.connecting_line = None
            child_node_a.right.connecting_line = None
            # TODO: re-merge, but blacklist current pair
            merge(child_node_a.left, child_node_a.right, blacklist=blacklist.append(()))
        else:
            print('Child A is a leaf!')
        if len(child_node_a.tracts) > 3:
            child_node_b.graph = []
            child_node_b.heads = []
            child_node_b.left.connecting_line = None
            child_node_b.right.connecting_line = None
            # TODO: re-merge, but blacklist current pair
            merge(child_node_b.left, child_node_b.right)
        else:
            print('Child B is a leaf!')

        # re-do visbility check
        merge_pairs = [(a, b) for a in child_node_a.heads for b in child_node_b.heads]

        final_merge_pairs = []
        for p in merge_pairs:
            p0_coords = [tract_coords[tract_coords.tract_ID == p[0]].X.values[0],
                         tract_coords[tract_coords.tract_ID == p[0]].Y.values[0]]
            p1_coords = [tract_coords[tract_coords.tract_ID == p[1]].X.values[0],
                         tract_coords[tract_coords.tract_ID == p[1]].Y.values[0]]
            newLine = LineString([p0_coords, p1_coords])
            # construct ring using a.graph and b.graph and newLine, and see if it forms a ring
            # if so, remove it from list of candidates
            ring = MultiLineString(child_node_a.graph + child_node_b.graph + [newLine])
            if ring.is_simple:
                final_merge_pairs.append(p)

        merge_pairs = [p for p in final_merge_pairs]
    else:
        # copy lines and heads into parent, plus new line (randomly chosen from merge_pairs)
        if len(merge_pairs) > 1:
            choice = merge_pairs[np.random.choice(range(len(merge_pairs)))]
        else:
            choice = merge_pairs[0]
        # print("For", child_node_a.tracts, "and", child_node_b.tracts, "choosing", choice)
        source_coords = [tract_coords[tract_coords.tract_ID == choice[0]].X.values[0],
                         tract_coords[tract_coords.tract_ID == choice[0]].Y.values[0]]
        dest_coords = [tract_coords[tract_coords.tract_ID == choice[1]].X.values[0],
                       tract_coords[tract_coords.tract_ID == choice[1]].Y.values[0]]
        connection = LineString([source_coords, dest_coords])
        child_node_a.connecting_line = connection
        child_node_b.connecting_line = connection
        child_node_a.parent.graph = child_node_a.graph + child_node_b.graph
        child_node_a.parent.graph.append(connection)
        # heads for parent are the 2 nodes that weren't connected
        if len(child_node_a.heads) == 1:
            child_node_a.parent.heads.append(child_node_a.heads[0])
        else:
            child_node_a.parent.heads.append(child_node_a.heads[1 - child_node_a.heads.index(choice[0])])
        if len(child_node_b.heads) == 1:
            child_node_a.parent.heads.append(child_node_b.heads[0])
        else:
            child_node_a.parent.heads.append(child_node_b.heads[1 - child_node_b.heads.index(choice[1])])
    # return parent as TreeNode object
    # print("Merged! Returning", child_node_a.parent.tracts, "/", child_node_b.parent.tracts)
    child_node_a.parent.graph_node()
    return child_node_a.parent  # or child_node_b.parent, it really doesn't matter

def merge_fix(node_a, node_b):
    """
    Fixes a merge issue when trapped case is detected
    Args:
        node_a:
        node_b:

    Returns:

    """

def merge_tree(root):
    """
    Main function for merge, runs it recursively.
    Args:
        root - root node of subtree
    Returns:
        TreeNode object, with lines and head/tail listed out inside
        None, if graph is blank/only has root node
    """

    # node is a leaf node
    if root.right == None and root.left == None:
        return root
    else:
        # left has an unmerged subtree
        if len(root.left.heads) == 0 and len(root.right.heads) > 0:
            return merge(merge_tree(root.left), root.right)
        # right has an unmerged subtree
        elif len(root.left.heads) > 0 and len(root.right.heads) == 0:
            return merge(root.left, merge_tree(root.right))
        # both children have unmerged subtrees
        elif len(root.left.heads) == 0 and len(root.right.heads) == 0:
            return merge(merge_tree(root.left), merge_tree(root.right))
        # both children exist and have merged subtrees, or are leaves
        else:
            return merge(root.left, root.right)


def clustering(tree, tracts):
    global tract_coords
    if (len(tracts) < 3):
        return 0
    else:
        # split into 2 groups using clustering
        tracts_to_split = tract_coords.loc[tract_coords.tract_ID.isin(tracts)]
        clusters = KMeans(n_clusters=2).fit_predict(
            np.column_stack([tracts_to_split.X.values, tracts_to_split.Y.values]))
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
# tract_coords = pd.read_csv('graph_tests/normal_30.csv')


def main(i=None):
    # print(tract_coords.head())
    pd.set_option('mode.chained_assignment', None)
    cluster_tree = assemble_tree()
    root_node = merge_tree(cluster_tree.root)
    root_node.graph_node()
    # print(root_node.heads)
    if len(root_node.heads) == 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    volume = 500
    actual = 0
    # print("Generating {} graphs and saving to 16-points-graphs/...".format(volume))
    # for i in tqdm(range(volume)):
    #     try:
    #         actual += main(i + 1)
    #     except RuntimeError:
    #         actual -= 1
    #         pass
    # print("Generation finished, {} generated, {}% accuracy.".format(actual, (actual / volume) * 100))
    print(main())
