"""
Principal Component Analysis (PCA) Implementation
Dimensionality reduction via eigendecomposition
"""

import numpy as np

class PCAScratch:
    """PCA from scratch using eigendecomposition."""
    
    def __init__(self, n_components=None, explained_variance_threshold=0.95):
        self.n_components = n_components
        self.explained_variance_threshold = explained_variance_threshold
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ratio_ = None
        self.mean_ = None
        
    def fit(self, X):
        """Fit PCA by computing eigenvectors of covariance matrix."""
        m, n = X.shape
        
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_ratio = np.cumsum(explained_variance_ratio)
        
        # Determine n_components
        if self.n_components is None:
            if self.explained_variance_threshold < 1:
                self.n_components = np.argmax(cumulative_ratio >= self.explained_variance_threshold) + 1
            else:
                self.n_components = n
        
        # Store results
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components]
        self.cumulative_explained_variance_ratio_ = cumulative_ratio[:self.n_components]
        
        return self
    
    def transform(self, X):
        """Project data onto principal components."""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Reconstruct original data from transformed."""
        return X_transformed @ self.components_ + self.mean_


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA as SklearnPCA
    
    X, _ = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    
    pca_custom = PCAScratch(n_components=3)
    X_transformed_custom = pca_custom.fit_transform(X)
    
    pca_sklearn = SklearnPCA(n_components=3)
    X_transformed_sklearn = pca_sklearn.fit_transform(X)
    
    print(f"Custom PCA shape: {X_transformed_custom.shape}")
    print(f"Sklearn PCA shape: {X_transformed_sklearn.shape}")
    print(f"Explained variance ratio: {pca_custom.explained_variance_ratio_}")
    print(f"Cumulative: {pca_custom.cumulative_explained_variance_ratio_[-1]:.3f}")
