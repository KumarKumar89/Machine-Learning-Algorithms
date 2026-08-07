"""
K-Means Clustering Implementation Tests
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import sys
sys.path.insert(0, '/workspace/ml-algorithms-deep-dive/k-means/src')

# Import implementations
from decision_trees.src.tree_node import DecisionTreeNode  # Placeholder for structure

def test_kmeans_initialization():
    """Test K-Means++ initialization spreads centroids."""
    from k_means.src.kmeans import KMeansVectorized
    
    X = np.random.randn(100, 2)
    km = KMeansVectorized(n_clusters=5, random_state=42)
    centroids = km._initialize_centroids(X)
    
    assert centroids.shape == (5, 2), "Centroids shape incorrect"
    assert len(np.unique(centroids.flatten())) > 1, "Centroids not diverse"
    print("✓ K-Means++ initialization")

def test_cluster_assignment():
    """Test points assigned to nearest centroid."""
    from k_means.src.kmeans import KMeansVectorized
    
    X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
    km = KMeansVectorized(n_clusters=2)
    km.centroids = np.array([[0.5, 0.5], [10.5, 10.5]])
    
    labels = km.predict(X)
    assert labels[0] == labels[1], "Close points should be in same cluster"
    assert labels[2] == labels[3], "Close points should be in same cluster"
    assert labels[0] != labels[2], "Distant points should be in different clusters"
    print("✓ Cluster assignment logic")

def test_centroid_update():
    """Test centroids move to mean of assigned points."""
    from k_means.src.kmeans import KMeansVectorized
    
    X = np.array([[0, 0], [2, 2], [10, 10], [12, 12]])
    km = KMeansVectorized(n_clusters=2)
    km.centroids = np.array([[0, 0], [10, 10]])
    
    labels = np.array([0, 0, 1, 1])
    new_centroids = km._update_centroids(X, labels)
    
    expected = np.array([[1, 1], [11, 11]])
    assert np.allclose(new_centroids, expected), "Centroids not updated to mean"
    print("✓ Centroid update (M-step)")

def test_inertia_computation():
    """Test inertia (within-cluster sum of squares)."""
    from k_means.src.kmeans import KMeansVectorized
    
    X = np.array([[0, 0], [1, 1], [10, 10]])
    km = KMeansVectorized(n_clusters=2)
    km.centroids = np.array([[0.5, 0.5], [10, 10]])
    labels = np.array([0, 0, 1])
    
    inertia = km._compute_inertia(X, labels)
    expected = 0.5 + 0.5 + 0  # Sum of squared distances
    assert abs(inertia - expected) < 0.01, f"Inertia incorrect: {inertia} vs {expected}"
    print("✓ Inertia computation")

def test_convergence():
    """Test algorithm converges."""
    from k_means.src.kmeans import KMeansVectorized
    
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    km = KMeansVectorized(n_clusters=3, random_state=42, tol=1e-6)
    km.fit(X)
    
    assert km.n_iter_ < 300, "Did not converge quickly"
    assert km.inertia_ > 0, "Inertia should be positive"
    print("✓ Convergence behavior")

def test_accuracy_vs_sklearn():
    """Test accuracy matches scikit-learn."""
    from k_means.src.kmeans import KMeansVectorized
    
    X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
    
    km_custom = KMeansVectorized(n_clusters=4, random_state=42)
    km_custom.fit(X)
    
    km_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_sklearn.fit(X)
    
    # Both should find similar structure
    score_custom = adjusted_rand_score(y_true, km_custom.labels_)
    score_sklearn = adjusted_rand_score(y_true, km_sklearn.labels_)
    
    assert score_custom > 0.7, f"Custom implementation ARI too low: {score_custom}"
    assert abs(score_custom - score_sklearn) < 0.1, "Significant difference from sklearn"
    print(f"✓ Accuracy vs sklearn (ARI: {score_custom:.3f} vs {score_sklearn:.3f})")

def test_different_k_values():
    """Test with different numbers of clusters."""
    from k_means.src.kmeans import KMeansVectorized
    
    X, _ = make_blobs(n_samples=200, centers=5, random_state=42)
    
    for k in [2, 3, 5, 8]:
        km = KMeansVectorized(n_clusters=k, random_state=42)
        km.fit(X)
        assert len(np.unique(km.labels_)) <= k, f"Too many clusters for K={k}"
    
    print("✓ Different K values")

def test_high_dimensional_data():
    """Test with high-dimensional data."""
    from k_means.src.kmeans import KMeansVectorized
    
    X = np.random.randn(100, 50)  # 50 features
    km = KMeansVectorized(n_clusters=3, random_state=42)
    km.fit(X)
    
    assert km.centroids.shape == (3, 50), "Centroid shape wrong for high-D"
    print("✓ High-dimensional data handling")

def test_empty_cluster_handling():
    """Test handling of empty clusters."""
    from k_means.src.kmeans import KMeansVectorized
    
    X = np.array([[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]])
    km = KMeansVectorized(n_clusters=4, random_state=42)
    
    # Force bad initialization that might create empty cluster
    km.centroids = np.array([[0, 0], [0.1, 0.1], [10, 10], [100, 100]])
    labels = km.predict(X)
    
    # Should handle empty cluster gracefully
    new_centroids = km._update_centroids(X, labels)
    assert new_centroids.shape == (4, 2), "Centroid shape wrong after update"
    print("✓ Empty cluster handling")

def test_predict_new_data():
    """Test prediction on unseen data."""
    from k_means.src.kmeans import KMeansVectorized
    
    X_train = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
    X_test = np.array([[0.5, 0.5], [10.5, 10.5]])
    
    km = KMeansVectorized(n_clusters=2, random_state=42)
    km.fit(X_train)
    
    test_labels = km.predict(X_test)
    assert len(test_labels) == 2, "Wrong number of predictions"
    assert test_labels[0] != test_labels[1], "Test points should be in different clusters"
    print("✓ Prediction on new data")

if __name__ == "__main__":
    print("Running K-Means Tests...\n")
    
    tests = [
        test_kmeans_initialization,
        test_cluster_assignment,
        test_centroid_update,
        test_inertia_computation,
        test_convergence,
        test_accuracy_vs_sklearn,
        test_different_k_values,
        test_high_dimensional_data,
        test_empty_cluster_handling,
        test_predict_new_data
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {str(e)}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{passed+failed} tests passed ({100*passed/(passed+failed):.1f}%)")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! K-Means implementation is correct.")
    else:
        print(f"⚠️  {failed} test(s) failed.")
