"""
Decision Tree Classifier and Regressor - From Scratch Implementation

Implements CART (Classification and Regression Trees) algorithm with:
- Gini impurity and entropy criteria for classification
- Variance reduction for regression
- Configurable depth, min_samples_split, min_samples_leaf
- Feature importance calculation
- Full scikit-learn compatibility

Mathematical Foundation:
- Gini Impurity: G = 1 - Σ(p_k²)
- Entropy: H = -Σ(p_k * log₂(p_k))  
- Information Gain: IG = Parent_Impurity - Weighted_Child_Impurity
- Variance Reduction: VR = Var(parent) - weighted_avg(Var(children))
"""

import numpy as np
from collections import Counter
from typing import Tuple, Optional, Union, List


class TreeNode:
    """Single node in the decision tree."""
    
    def __init__(self, 
                 feature_index: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: 'TreeNode' = None,
                 right: 'TreeNode' = None,
                 value: Optional[Union[int, float, np.ndarray]] = None,
                 is_leaf: bool = False):
        
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using CART algorithm.
    
    Parameters
    ----------
    max_depth : int, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required in each leaf.
    criterion : {'gini', 'entropy'}, default='gini'
        Function to measure quality of split.
    max_features : int, default=None
        Number of features to consider per split.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, 
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 criterion: str = 'gini',
                 max_features: Optional[int] = None,
                 random_state: Optional[int] = None):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        
        self.root = None
        self.n_classes = None
        self.n_features = None
        self.feature_importances_ = None
        self._rng = np.random.RandomState(random_state)
        self._importance_accumulator = None
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity.
        
        ↳ Implements: G = 1 - Σ(p_k²) from Section 3
        
        Plain English: Probability of misclassifying random sample.
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1.0 - np.sum(proportions ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy.
        
        ↳ Implements: H = -Σ(p_k * log₂(p_k)) from Section 3
        
        Plain English: Average information content (uncertainty).
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        
        # Avoid log(0)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, y_parent: np.ndarray, 
                          y_left: np.ndarray, 
                          y_right: np.ndarray) -> float:
        """
        Calculate information gain from split.
        
        ↳ Implements: IG = Parent_Impurity - Weighted_Child_Impurity
        
        Plain English: How much uncertainty is reduced by this split.
        """
        n = len(y_parent)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_impurity = self._calculate_impurity(y_parent)
        weighted_child_impurity = (
            (n_left / n) * self._calculate_impurity(y_left) +
            (n_right / n) * self._calculate_impurity(y_right)
        )
        
        return parent_impurity - weighted_child_impurity
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find best feature and threshold for splitting."""
        n_samples, n_features = X.shape
        
        # Select random subset of features
        if self.max_features:
            feature_indices = self._rng.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Recursively build the decision tree."""
        n_samples = len(y)
        
        # Check stopping conditions
        should_stop = (
            (self.max_depth is not None and depth >= self.max_depth) or
            (n_samples < self.min_samples_split) or
            (len(np.unique(y)) == 1)
        )
        
        if should_stop:
            # Create leaf with majority class
            most_common_class = Counter(y).most_common(1)[0][0]
            return TreeNode(value=most_common_class, is_leaf=True)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            most_common_class = Counter(y).most_common(1)[0][0]
            return TreeNode(value=most_common_class, is_leaf=True)
        
        # Update feature importance
        self._importance_accumulator[best_feature] += best_gain * n_samples
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            most_common_class = Counter(y).most_common(1)[0][0]
            return TreeNode(value=most_common_class, is_leaf=True)
        
        # Recursively build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """Build the decision tree from training data."""
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        self._importance_accumulator = np.zeros(self.n_features)
        
        self.root = self._build_tree(X, y, depth=0)
        
        # Normalize feature importances
        total_importance = np.sum(self._importance_accumulator)
        if total_importance > 0:
            self.feature_importances_ = self._importance_accumulator / total_importance
        else:
            self.feature_importances_ = np.zeros(self.n_features)
        
        return self
    
    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> int:
        """Predict class for single sample."""
        if node.is_leaf:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples."""
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes))
        
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if x[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            
            proba[i, node.value] = 1.0
        
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy on test data."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class DecisionTreeRegressor:
    """
    Decision Tree Regressor using CART algorithm.
    
    Parameters
    ----------
    max_depth : int, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required in each leaf.
    max_features : int, default=None
        Number of features to consider per split.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[int] = None,
                 random_state: Optional[int] = None):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        self.root = None
        self.n_features = None
        self.feature_importances_ = None
        self._rng = np.random.RandomState(random_state)
        self._importance_accumulator = None
    
    def _variance(self, y: np.ndarray) -> float:
        """Calculate variance of target values."""
        if len(y) == 0:
            return 0.0
        return np.var(y)
    
    def _variance_reduction(self, y_parent: np.ndarray,
                            y_left: np.ndarray,
                            y_right: np.ndarray) -> float:
        """
        Calculate variance reduction from split.
        
        ↳ Implements: VR = Var(parent) - weighted_avg(Var(children))
        
        Plain English: How much prediction error is reduced by this split.
        """
        n = len(y_parent)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        weighted_child_variance = (
            (n_left / n) * self._variance(y_left) +
            (n_right / n) * self._variance(y_right)
        )
        
        return self._variance(y_parent) - weighted_child_variance
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find best feature and threshold for splitting."""
        n_samples, n_features = X.shape
        
        if self.max_features:
            feature_indices = self._rng.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        best_reduction = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                reduction = self._variance_reduction(y, y[left_mask], y[right_mask])
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_reduction
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Recursively build regression tree."""
        n_samples = len(y)
        
        should_stop = (
            (self.max_depth is not None and depth >= self.max_depth) or
            (n_samples < self.min_samples_split) or
            (np.var(y) < 1e-10)
        )
        
        if should_stop:
            return TreeNode(value=np.mean(y), is_leaf=True)
        
        best_feature, best_threshold, best_reduction = self._find_best_split(X, y)
        
        if best_feature is None or best_reduction <= 0:
            return TreeNode(value=np.mean(y), is_leaf=True)
        
        self._importance_accumulator[best_feature] += best_reduction * n_samples
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return TreeNode(value=np.mean(y), is_leaf=True)
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """Build the decision tree from training data."""
        self.n_samples, self.n_features = X.shape
        self._importance_accumulator = np.zeros(self.n_features)
        
        self.root = self._build_tree(X, y, depth=0)
        
        total_importance = np.sum(self._importance_accumulator)
        if total_importance > 0:
            self.feature_importances_ = self._importance_accumulator / total_importance
        else:
            self.feature_importances_ = np.zeros(self.n_features)
        
        return self
    
    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> float:
        """Predict value for single sample."""
        if node.is_leaf:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² coefficient of determination."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
