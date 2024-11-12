from collections import Counter
import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            x_samples, y_samples = self._bootstrap_samples(x, y)
            tree.fit(x_samples, y_samples)
            self.trees.append(tree)

    def _bootstrap_samples(self, x, y):
        n_samples = x.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return x[idxs, y[idxs]]
    
    def _most_common_label(self, y):
        counter = Counter()
        most_common = counter.most_common(1)[0][0]

        return most_common

    def predict(self, x):
        predictions = np.array(
            [tree.predict[x] for tree in self.trees]
        )
        tree_preds = np.swapaxes(predictions, 0, 1)
        
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])

        return predictions