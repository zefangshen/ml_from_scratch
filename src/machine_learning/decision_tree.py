import numpy as np
from collections import Counter

"""Notes: Decision tree for classification
- The key it to find the best split at each node, based on information gain
    - the dataset is splitted in different branches. 
    - based on what feature and what value to split the dataset
    - the information gain measures the uncertainty/randomness in y before and 
        after splitting. Splitting should reduced uncertainty 
    - Other metric other than information gain could be used 
- Recursive function _grow_tree() to create nodes at different depths.
- A Node needs the index of the feature used for splitting, the threshold for 
    splitting, value if leaf_node, or left and right child nodes.

"""

class Node:
    def __init__(
            self, feature=None, threshold=None, left=None, right=None, *,
            value=None
        ):
        # keyword only arguments after *

        self.feature = feature  # feature that used to split the node
        self.threshold = threshold  # threshold of the feature for the split
        self.left = left  # left child node
        self.right = right  # right child node
        self.value = None  # output value if it is a leaf node
    
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split  # minimum number of samples in splits
        self.max_depth = max_depth  # maximum depths of the tree
        self.n_features = n_features  # n_features used for the split
        self.root = None  # root node of the tree

    def fit(self, x, y):
        # number of features should not be larger than total number of features,
        # or the n_features specified
        self.n_features = x.shape[1] if not self.n_features else \
            min(self.n_features, x.shape[1])  

        # create a decision tree, the _grow_tree() is a recursive function
        # it stops when stopping criteria are reached 
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        # if stops
        if (
            depth >= self.max_depth or n_labels==1 or n_samples < \
                self.min_samples_split
        ):
            # each leaf node output a class, using majority vote  
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # find the best split
        #  randomly select a few features to split the dataset
        feat_ids = np.random.choice(n_features, self.n_features, replace=False)
        #  loop through the features and find out the best splitting scheme 
        best_feature, best_thresh = self._best_split(x, y, feat_ids)

        # create child node
        #  create child nodes based the above best split 
        left_ids, right_ids = self._split(x[:, best_feature], best_thresh)
        #  grow the tree on the left side
        left = self._grow_tree(x[left_ids, :], y[left_ids], depth+1)
        #  grow the tree on the right side
        right = self._grow_tree(x[right_ids, :], y[right_ids], depth+1)

        return Node(best_feature, best_thresh, left, right)


    def _most_common_label(self, y):
        # majority vote, find the most frequent class
        counter = Counter(y)
        value = counter.most_common(1)[0][0]

        return value

    def _best_split(self, x, y, feat_ids):
        # find the best split based on information gain

        # initialisation
        best_gain = -1
        split_idx, split_threshold = None, None
        
        # loop through all features
        for feat_idx in feat_ids:
            # get the feature column
            x_col = x[:, feat_idx]
            # get all position threshold from the feature column
            thresholds = np.unique(x_col)

            # loop through all 
            for threshold in thresholds: 
                # calculate the information gain for a specific feature with a 
                #  specific threshold
                gain = self._information_gain(y, x_col, threshold)

                # update the gain  if a better one is found
                if gain > best_gain:
                    best_gain = gain 
                    # record the best feature and threshold
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold


    def _information_gain(self, y, x_col, threshold):
        # calculate the information gain
        #  parent entropy - weighted sum of children entropy

        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_ids, right_ids = self._split(x_col, threshold)

        # same as parent
        if len(left_ids) == 0 or len(right_ids) == 0:
            return 0 

        # calculate the weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_ids), len(right_ids)
        # calculate left and right child entropy
        e_l, e_r = self._entropy(y[left_ids]), self._entropy(y[right_ids])
        # calculate weighted sum of the children entropy
        children_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - children_entropy 

        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _split(self, x_col, threshold):
        # get the indices of for left and right child in a node split

        left_ids = np.argwhere(x_col <= threshold).flatten()
        right_ids = np.argwhere(x_col > threshold).flatten()

        return left_ids, right_ids

    def predict(self, x):
        # prediction for each sample
        return np.array([self._traverse_tree(xi, self.root) for xi in x] )
    
    def _traverse_tree(self, x, node):
        # travel across the tree until leaf node 

        # if leaf node return prediction
        if node.is_leaf_node():
            return node.value()

        # if left node, traverse the left node
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        # otherwise traverse right node
        return self._traverse_tree(x, node.right)