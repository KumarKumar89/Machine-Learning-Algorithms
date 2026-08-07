"""
Random Forest Implementation
Ensemble of decision trees with bagging and feature randomness
"""

import numpy as np

class DecisionTreeStump:
    """Simple decision tree for Random Forest."""
    
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
        
    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)
    
    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None, None
        
        # Random feature subset
        n_features = int(np.sqrt(n)) if self.max_features is None else min(self.max_features, n)
        feature_idx = np.random.choice(n, n_features, replace=False)
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feat in feature_idx:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gini = (np.sum(left_mask) * self._gini(y[left_mask]) + 
                       np.sum(right_mask) * self._gini(y[right_mask])) / m
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feat
                    best_threshold = thresh
        
        return best_feature, best_threshold, best_gini
    
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return {'leaf': True, 'prediction': np.bincount(y).argmax() if len(y) > 0 else 0}
        
        feat, thresh, _ = self._best_split(X, y)
        
        if feat is None:
            return {'leaf': True, 'prediction': np.bincount(y).argmax()}
        
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        
        return {
            'leaf': False,
            'feature': feat,
            'threshold': thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        if node['leaf']:
            return node['prediction']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForestClassifier:
    """Random Forest Classifier with bagging."""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        m, n = X.shape
        self.classes_ = np.unique(y)
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(m, m, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train tree
            tree = DecisionTreeStump(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        # Majority voting
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier as SklearnRF
    
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8, random_state=42)
    
    rf_custom = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf_custom.fit(X, y)
    
    rf_sklearn = SklearnRF(n_estimators=50, max_depth=5, random_state=42)
    rf_sklearn.fit(X, y)
    
    acc_custom = rf_custom.score(X, y)
    acc_sklearn = rf_sklearn.score(X, y)
    
    print(f"Custom RF Accuracy: {acc_custom:.3f}")
    print(f"Sklearn RF Accuracy: {acc_sklearn:.3f}")
    print(f"Difference: {abs(acc_custom - acc_sklearn):.3f}")
