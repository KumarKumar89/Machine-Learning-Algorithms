"""
K-Means Clustering Implementation
Both educational (loop-based) and production (vectorized) versions
"""

import numpy as np

class KMeansScratch:
    """
    K-Means Clustering from scratch using NumPy.
    
    Mathematical foundation:
    - Objective: min Σ||x⁽ⁱ⁾ - μ_c⁽ⁱ⁾||²
    - E-step: c⁽ⁱ⁾ := argmin_k ||x⁽ⁱ⁾ - μ_k||²
    - M-step: μ_k := mean of points in cluster k
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, init='kmeans++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X):
        """Initialize centroids using K-Means++ algorithm."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        m, n = X.shape
        
        if self.init == 'random':
            indices = np.random.choice(m, self.n_clusters, replace=False)
            return X[indices]
        
        elif self.init == 'kmeans++':
            centroids = []
            first_idx = np.random.randint(m)
            centroids.append(X[first_idx].copy())
            
            for _ in range(1, self.n_clusters):
                distances = np.zeros(m)
                for i in range(m):
                    min_dist = float('inf')
                    for c in centroids:
                        dist = np.sum((X[i] - c) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                    distances[i] = min_dist
                
                probabilities = distances / np.sum(distances)
                cumulative_probs = np.cumsum(probabilities)
                r = np.random.rand()
                next_idx = np.searchsorted(cumulative_probs, r)
                centroids.append(X[next_idx].copy())
            
            return np.array(centroids)
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
    
    def _assign_clusters(self, X):
        """Assign each point to nearest centroid (E-step)."""
        m = X.shape[0]
        labels = np.zeros(m, dtype=int)
        
        for i in range(m):
            distances = np.zeros(self.n_clusters)
            for k in range(self.n_clusters):
                distances[k] = np.sum((X[i] - self.centroids[k]) ** 2)
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """Update centroids to mean of assigned points (M-step)."""
        n = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n))
        
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                new_centroids[k] = X[np.random.randint(X.shape[0])]
        
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """Compute within-cluster sum of squares."""
        inertia = 0.0
        for i in range(len(X)):
            inertia += np.sum((X[i] - self.centroids[labels[i]]) ** 2)
        return inertia
    
    def fit(self, X):
        """Fit K-Means clustering."""
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)
            
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        
        self.n_iter_ = iteration + 1
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        return self._assign_clusters(X)


class KMeansVectorized:
    """Fully vectorized K-Means using broadcasting."""
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def _initialize_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        
        m = X.shape[0]
        centroids = []
        first_idx = np.random.randint(m)
        centroids.append(X[first_idx].copy())
        
        for _ in range(1, self.n_clusters):
            distances = np.zeros(m)
            for c in centroids:
                dist = np.sum((X - c) ** 2, axis=1)
                distances = np.minimum(distances, dist)
            
            total = float(np.sum(distances))
            if total <= 0 or not np.isfinite(total):
                # Degenerate case (e.g. duplicated points): sample uniformly.
                next_idx = int(rng.integers(m))
            else:
                probs = distances / total
                next_idx = int(rng.choice(m, p=probs))
            centroids.append(X[next_idx].copy())
        
        return np.array(centroids)
    
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            distances = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[k] = X[np.random.randint(X.shape[0])]
            
            shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        
        self.n_iter_ = iteration + 1
        self.labels_ = labels
        self.inertia_ = np.sum((X - self.centroids[labels]) ** 2)
        
        return self
    
    def predict(self, X):
        distances = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Helper for testing."""
        n = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n))
        
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                new_centroids[k] = X[np.random.randint(X.shape[0])]
        
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """Helper for testing."""
        return np.sum((X - self.centroids[labels]) ** 2)
