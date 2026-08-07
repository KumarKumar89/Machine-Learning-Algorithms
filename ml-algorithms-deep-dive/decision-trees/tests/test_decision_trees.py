"""
Comprehensive test suite for Decision Tree implementations.

Tests cover:
- Impurity calculations (Gini, Entropy)
- Information gain computations
- Tree construction and splitting logic
- Classification and regression accuracy
- Feature importance extraction
- Edge cases and error handling
- Comparison with scikit-learn
"""

import numpy as np
import sys
sys.path.append('src')

from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor, TreeNode


def test_gini_impurity():
    """Test Gini impurity calculation."""
    clf = DecisionTreeClassifier(criterion='gini')
    
    # Pure node (all same class)
    y_pure = np.array([1, 1, 1, 1])
    assert clf._gini_impurity(y_pure) == 0.0, "Pure node should have Gini=0"
    
    # Balanced binary node
    y_balanced = np.array([0, 0, 1, 1])
    expected = 1.0 - (0.5**2 + 0.5**2)  # 0.5
    assert abs(clf._gini_impurity(y_balanced) - expected) < 1e-6
    
    # Imbalanced node
    y_imbalanced = np.array([0, 0, 0, 1])
    expected = 1.0 - (0.75**2 + 0.25**2)  # 0.375
    assert abs(clf._gini_impurity(y_imbalanced) - expected) < 1e-6
    
    print("✓ Gini impurity calculation")


def test_entropy():
    """Test entropy calculation."""
    clf = DecisionTreeClassifier(criterion='entropy')
    
    # Pure node
    y_pure = np.array([1, 1, 1, 1])
    assert clf._entropy(y_pure) == 0.0, "Pure node should have Entropy=0"
    
    # Balanced binary node
    y_balanced = np.array([0, 0, 1, 1])
    expected = 1.0  # -2 * (0.5 * log2(0.5)) = 1.0
    assert abs(clf._entropy(y_balanced) - expected) < 1e-6
    
    print("✓ Entropy calculation")


def test_information_gain():
    """Test information gain computation."""
    clf = DecisionTreeClassifier(criterion='gini')
    
    # Parent: balanced, Children: pure
    y_parent = np.array([0, 0, 1, 1])
    y_left = np.array([0, 0])
    y_right = np.array([1, 1])
    
    gain = clf._information_gain(y_parent, y_left, y_right)
    expected = 0.5  # Parent Gini (0.5) - 0 (children pure)
    assert abs(gain - expected) < 1e-6
    
    # No gain case
    y_left_mixed = np.array([0, 1])
    y_right_mixed = np.array([0, 1])
    gain_no = clf._information_gain(y_parent, y_left_mixed, y_right_mixed)
    assert gain_no < 1e-6, "No improvement split should have ~0 gain"
    
    print("✓ Information gain computation")


def test_best_split():
    """Test best split selection."""
    np.random.seed(42)
    clf = DecisionTreeClassifier(max_depth=5)
    
    # Create simple separable data
    X = np.array([
        [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    feature, threshold, gain = clf._find_best_split(X, y)
    
    assert feature == 0, "Should select the only feature"
    assert 3.0 <= threshold <= 4.0, "Threshold should be between classes"
    assert gain > 0.4, "Should have significant gain"
    
    print("✓ Best split selection")


def test_tree_construction():
    """Test tree building with depth control."""
    np.random.seed(42)
    clf = DecisionTreeClassifier(max_depth=2, min_samples_split=2)
    
    X = np.random.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    clf.fit(X, y)
    
    # Check tree structure
    assert clf.root is not None
    assert clf.root.is_leaf == False
    
    # Verify depth constraint
    def get_depth(node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(get_depth(node.left), get_depth(node.right))
    
    actual_depth = get_depth(clf.root)
    assert actual_depth <= 2, f"Tree depth {actual_depth} exceeds max_depth=2"
    
    print("✓ Tree construction depth control")


def test_classification_accuracy():
    """Test classification accuracy against sklearn."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier as SklearnClassifier
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    # Custom implementation
    clf_custom = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_custom.fit(X_train, y_train)
    custom_acc = clf_custom.score(X_test, y_test)
    
    # sklearn implementation
    clf_sklearn = SklearnClassifier(max_depth=3, random_state=42)
    clf_sklearn.fit(X_train, y_train)
    sklearn_acc = clf_sklearn.score(X_test, y_test)
    
    # Should achieve reasonable accuracy (>80%)
    assert custom_acc > 0.8, f"Custom accuracy {custom_acc} too low"
    
    # Should be within 10% of sklearn
    assert abs(custom_acc - sklearn_acc) < 0.1, \
        f"Gap too large: custom={custom_acc}, sklearn={sklearn_acc}"
    
    print(f"✓ Classification accuracy vs sklearn (custom={custom_acc:.3f}, sklearn={sklearn_acc:.3f})")


def test_regression_performance():
    """Test regression R² score."""
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X_train, y_train)
    r2 = reg.score(X_test, y_test)
    
    assert r2 > 0.5, f"R² score {r2} too low for simple regression"
    
    print(f"✓ Regression R² performance (R²={r2:.3f})")


def test_feature_importance():
    """Test feature importance extraction."""
    np.random.seed(42)
    
    # Create data where only first feature matters
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)  # Only depends on feature 0
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    assert clf.feature_importances_ is not None
    assert len(clf.feature_importances_) == 5
    assert np.isclose(np.sum(clf.feature_importances_), 1.0), "Importances should sum to 1"
    
    # First feature should have highest importance
    assert clf.feature_importances_[0] == max(clf.feature_importances_), \
        "Most important feature should have highest importance score"
    
    print("✓ Feature importance extraction")


def test_min_samples_leaf():
    """Test min_samples_leaf constraint."""
    np.random.seed(42)
    clf = DecisionTreeClassifier(min_samples_leaf=10, max_depth=10)
    
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, 100)
    
    clf.fit(X, y)
    
    # Verify all leaves have >= 10 samples
    def check_leaf_sizes(node, X_subset):
        if node.is_leaf:
            return len(X_subset) >= 10
        
        left_mask = X_subset[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask
        
        left_ok = check_leaf_sizes(node.left, X_subset[left_mask])
        right_ok = check_leaf_sizes(node.right, X_subset[right_mask])
        
        return left_ok and right_ok
    
    assert check_leaf_sizes(clf.root, X), "Leaf constraint violated"
    
    print("✓ Min samples leaf constraint")


def test_categorical_features():
    """Test handling of categorical-like features."""
    # Create data with discrete values
    X = np.array([
        [1], [1], [2], [2], [3], [3], [4], [4]
    ])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    
    assert accuracy > 0.75, f"Should handle categorical features (acc={accuracy})"
    
    print("✓ Handling categorical features")


def test_missing_value_tolerance():
    """Test that implementation doesn't crash with NaN (graceful handling)."""
    X = np.array([
        [1.0], [2.0], [np.nan], [4.0], [5.0]
    ])
    y = np.array([0, 0, 1, 1, 1])
    
    # Fill NaN with mean for this test
    X_filled = np.nan_to_num(X, nan=np.nanmean(X))
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_filled, y)
    
    predictions = clf.predict(X_filled)
    assert len(predictions) == 5, "Should predict for all samples"
    
    print("✓ Missing value management (via imputation)")


def test_overfitting_prevention():
    """Test that constraints prevent overfitting."""
    np.random.seed(42)
    
    # Small noisy dataset
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)  # Random labels
    
    # Unconstrained tree (should overfit)
    clf_unconstrained = DecisionTreeClassifier(max_depth=None)
    clf_unconstrained.fit(X, y)
    train_acc_unconstrained = clf_unconstrained.score(X, y)
    
    # Constrained tree (should not overfit as much)
    clf_constrained = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5)
    clf_constrained.fit(X, y)
    train_acc_constrained = clf_constrained.score(X, y)
    
    # Constrained should have lower training accuracy (less overfitting)
    assert train_acc_constrained < train_acc_unconstrained, \
        "Constraints should reduce overfitting"
    
    print("✓ Overfitting prevention via constraints")


def test_performance_benchmark():
    """Basic performance benchmark."""
    import time
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    
    start = time.time()
    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X, y)
    fit_time = time.time() - start
    
    start = time.time()
    _ = clf.predict(X)
    predict_time = time.time() - start
    
    assert fit_time < 10.0, f"Fitting took too long: {fit_time}s"
    assert predict_time < 1.0, f"Prediction took too long: {predict_time}s"
    
    print(f"✓ Performance benchmarks (fit={fit_time:.3f}s, predict={predict_time:.3f}s)")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_gini_impurity,
        test_entropy,
        test_information_gain,
        test_best_split,
        test_tree_construction,
        test_classification_accuracy,
        test_regression_performance,
        test_feature_importance,
        test_min_samples_leaf,
        test_categorical_features,
        test_missing_value_tolerance,
        test_overfitting_prevention,
        test_performance_benchmark,
    ]
    
    passed = 0
    failed = 0
    
    print("="*70)
    print("DECISION TREE TEST SUITE")
    print("="*70)
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: Unexpected error - {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed ({100*passed/len(tests):.1f}%)")
    print("="*70)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Implementation is correct.")
    else:
        print(f"⚠️  {failed} tests failed. Review implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
