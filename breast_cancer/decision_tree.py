import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self,max_depth=10,min_samples_split=2,max_features=None):
        self.max_depth = max_depth
        self.min_samples_split=min_samples_split
        self.max_features = max_features
        self.tree = None 

    def gini(self,y):
        counts = np.array(list(Counter(y).values()))
        probabilities = counts/np.sum(counts)
        gini = 1 - np.sum(probabilities **2)
        return gini

    def best_split(self,X,y):
        n_samples ,  n_features = X.shape
        if self.max_features is None:
            features = np.arange(n_features)
        else:
            features = np.random.choice(
                n_features,min(self.max_features,n_features),replace=False
            )
        best_gain = -1
        best_split_info = None
        parent_gini = self.gini(y)

        for feature in features:
            thresholds = np.unique(X[:,features])

            for threshold in thresholds[:-1]:
                left_mask = X[:,feature] <= threshold
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                weighted_gini = (len(y_left)/n_samples) * self.gini(y_left) + (len(y_right)/n_samples) * self.gini(y_right)
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_split_info = (feature,threshold,left_mask,right_mask)

        return best_split_info,best_gain

    def build_tree(self,X,y,depth=0):

        if (depth>=self.max_depth or len(y)<self.min_samples_split or len(np.unique(y))==1):
            return Counter(y).most_common(1)[0][0]

        split, gain = self.best_split(X,y)

        if split is None or gain <= 0:
            return Counter(y).most_common(1)[0][0]

        feature, threshold, left_mask, right_mask = split

        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)

        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child
        }

    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        return self

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree

        if not isinstance(node, dict):
            return node

        feature = node["feature"]
        threshold = node["threshold"]

        if x[feature] <= threshold:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


class RandomForest:
    def __init__(self,n_trees=10,max_depth=10,min_samples_split=2,max_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.max_features is None:
            max_features = max(1, int(np.sqrt(n_features)))
        else:
            max_features = self.max_features

        self.trees = []

        for _ in range(self.n_trees):
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True
            )
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_predictions)