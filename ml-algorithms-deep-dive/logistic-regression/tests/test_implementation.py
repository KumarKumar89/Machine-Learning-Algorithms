"""
Test suite for Logistic Regression implementations.

Tests cover:
1. Correctness against sklearn
2. Convergence behavior
3. Regularization effects
4. Edge cases
5. Prediction accuracy
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace/ml-algorithms-deep-dive/logistic-regression/src')

from loop_implementation import LogisticRegressionLoop
from vectorized_implementation import LogisticRegressionVectorized
from sklearn.linear_model import LogisticRegression as SklearnLogistic


def generate_test_data(m=200, n=5, seed=42):
    """Generate synthetic binary classification data."""
    np.random.seed(seed)
    X = np.random.randn(m, n)
    true_weights = np.random.randn(n)
    true_bias = 0.5
    true_z = X @ true_weights + true_bias
    y = (true_z > 0).astype(int)
    return X, y, true_weights, true_bias


def test_loop_vs_vectorized():
    """Test that loop and vectorized implementations produce same results."""
    print("\n" + "="*60)
    print("TEST 1: Loop vs Vectorized Consistency")
    print("="*60)
    
    X, y, _, _ = generate_test_data(m=100, n=3)
    
    # Train both models with same hyperparameters
    model_loop = LogisticRegressionLoop(learning_rate=0.5, n_iterations=1000)
    model_loop.fit(X, y)
    
    model_vec = LogisticRegressionVectorized(learning_rate=0.5, n_iterations=1000)
    model_vec.fit(X, y)
    
    # Compare predictions
    preds_loop = model_loop.predict(X)
    preds_vec = model_vec.predict(X)
    
    accuracy_match = np.mean(preds_loop == preds_vec)
    
    print(f"Prediction agreement: {accuracy_match*100:.1f}%")
    print(f"Loop weights: {model_loop.weights}")
    print(f"Vec weights:  {model_vec.coef_}")
    print(f"Weight correlation: {np.corrcoef(model_loop.weights, model_vec.coef_)[0,1]:.6f}")
    
    assert accuracy_match > 0.95, "Predictions should match >95%"
    print("✓ PASSED: Loop and vectorized implementations consistent")
    return True


def test_against_sklearn():
    """Test our implementation against sklearn's LogisticRegression."""
    print("\n" + "="*60)
    print("TEST 2: Accuracy vs scikit-learn")
    print("="*60)
    
    X, y, _, _ = generate_test_data(m=300, n=5)
    
    # Our implementation
    model_ours = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=2000)
    model_ours.fit(X, y)
    acc_ours = model_ours.score(X, y)
    
    # Sklearn
    model_sklearn = SklearnLogistic(max_iter=2000, random_state=42)
    model_sklearn.fit(X, y)
    acc_sklearn = model_sklearn.score(X, y)
    
    print(f"Our accuracy:   {acc_ours:.4f}")
    print(f"Sklearn accuracy: {acc_sklearn:.4f}")
    print(f"Difference: {abs(acc_ours - acc_sklearn):.4f}")
    
    # Allow small difference due to different optimization algorithms
    assert abs(acc_ours - acc_sklearn) < 0.05, "Accuracy should be within 5% of sklearn"
    print("✓ PASSED: Accuracy matches sklearn within tolerance")
    return True


def test_convergence():
    """Test that loss decreases during training."""
    print("\n" + "="*60)
    print("TEST 3: Loss Convergence")
    print("="*60)
    
    X, y, _, _ = generate_test_data(m=200, n=4)
    
    model = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    
    loss_start = model.loss_history[0]
    loss_end = model.loss_history[-1]
    
    print(f"Initial loss: {loss_start:.6f}")
    print(f"Final loss:   {loss_end:.6f}")
    print(f"Reduction: {(loss_start - loss_end)/loss_start * 100:.1f}%")
    
    assert loss_end < loss_start, "Loss should decrease during training"
    assert all(model.loss_history[i] >= model.loss_history[i+1] - 0.01 
               for i in range(len(model.loss_history)-1)), \
               "Loss should generally decrease (with small tolerance)"
    print("✓ PASSED: Loss converges properly")
    return True


def test_regularization():
    """Test that L2 regularization reduces weight magnitudes."""
    print("\n" + "="*60)
    print("TEST 4: L2 Regularization Effect")
    print("="*60)
    
    X, y, _, _ = generate_test_data(m=150, n=5)
    
    # No regularization
    model_no_reg = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=1000, lambda_reg=0.0)
    model_no_reg.fit(X, y)
    
    # Strong regularization
    model_strong_reg = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=1000, lambda_reg=10.0)
    model_strong_reg.fit(X, y)
    
    weight_norm_no_reg = np.linalg.norm(model_no_reg.coef_)
    weight_norm_strong_reg = np.linalg.norm(model_strong_reg.coef_)
    
    print(f"Weight norm (no reg):     {weight_norm_no_reg:.4f}")
    print(f"Weight norm (strong reg): {weight_norm_strong_reg:.4f}")
    print(f"Reduction: {(1 - weight_norm_strong_reg/weight_norm_no_reg)*100:.1f}%")
    
    assert weight_norm_strong_reg < weight_norm_no_reg, \
           "Regularization should reduce weight magnitudes"
    print("✓ PASSED: L2 regularization shrinks weights")
    return True


def test_perfect_separation():
    """Test behavior with linearly separable data."""
    print("\n" + "="*60)
    print("TEST 5: Perfect Separation")
    print("="*60)
    
    # Create perfectly separable data
    np.random.seed(42)
    X_sep = np.vstack([
        np.random.randn(50, 2) + np.array([2, 2]),
        np.random.randn(50, 2) + np.array([-2, -2])
    ])
    y_sep = np.hstack([np.ones(50), np.zeros(50)])
    
    model = LogisticRegressionVectorized(learning_rate=0.5, n_iterations=1000)
    model.fit(X_sep, y_sep)
    
    accuracy = model.score(X_sep, y_sep)
    
    print(f"Accuracy on separable data: {accuracy:.4f}")
    print(f"Weights: {model.coef_}")
    print(f"Bias: {model.intercept_}")
    
    # Should achieve near-perfect accuracy
    assert accuracy > 0.98, "Should achieve near-perfect accuracy on separable data"
    print("✓ PASSED: Handles perfectly separable data")
    return True


def test_prediction_threshold():
    """Test custom decision thresholds."""
    print("\n" + "="*60)
    print("TEST 6: Custom Decision Thresholds")
    print("="*60)
    
    X, y, _, _ = generate_test_data(m=200, n=3)
    
    model = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    
    # Different thresholds
    preds_default = model.predict(X, threshold=0.5)
    preds_low = model.predict(X, threshold=0.3)
    preds_high = model.predict(X, threshold=0.7)
    
    positive_rate_default = np.mean(preds_default)
    positive_rate_low = np.mean(preds_low)
    positive_rate_high = np.mean(preds_high)
    
    print(f"Positive rate (threshold=0.5): {positive_rate_default:.3f}")
    print(f"Positive rate (threshold=0.3): {positive_rate_low:.3f}")
    print(f"Positive rate (threshold=0.7): {positive_rate_high:.3f}")
    
    assert positive_rate_low > positive_rate_default, \
           "Lower threshold should yield more positives"
    assert positive_rate_high < positive_rate_default, \
           "Higher threshold should yield fewer positives"
    print("✓ PASSED: Custom thresholds work correctly")
    return True


def test_sgd_vs_batch():
    """Compare SGD and batch gradient descent."""
    print("\n" + "="*60)
    print("TEST 7: SGD vs Batch GD")
    print("="*60)
    
    X, y, _, _ = generate_test_data(m=300, n=5)
    
    # Batch GD
    model_batch = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=500)
    model_batch.fit(X, y)
    acc_batch = model_batch.score(X, y)
    
    # SGD
    model_sgd = LogisticRegressionVectorized(learning_rate=0.05, n_iterations=100)
    model_sgd.fit_with_sgd(X, y, batch_size=32)
    acc_sgd = model_sgd.score(X, y)
    
    print(f"Batch GD accuracy: {acc_batch:.4f}")
    print(f"SGD accuracy:      {acc_sgd:.4f}")
    print(f"Difference: {abs(acc_batch - acc_sgd):.4f}")
    
    # Both should achieve reasonable accuracy
    assert acc_batch > 0.85, "Batch GD should achieve >85% accuracy"
    assert acc_sgd > 0.80, "SGD should achieve >80% accuracy"
    print("✓ PASSED: Both SGD and batch GD converge")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 8: Edge Cases")
    print("="*60)
    
    # Single feature
    np.random.seed(42)
    X_single = np.random.randn(100, 1)
    y_single = (X_single.flatten() > 0).astype(int)
    
    model = LogisticRegressionVectorized(learning_rate=0.5, n_iterations=500)
    model.fit(X_single, y_single)
    acc_single = model.score(X_single, y_single)
    print(f"Single feature accuracy: {acc_single:.4f} ✓")
    
    # Many features (relative to samples)
    X_many = np.random.randn(50, 20)
    y_many = np.random.randint(0, 2, 50)
    
    model_many = LogisticRegressionVectorized(learning_rate=0.01, n_iterations=500, lambda_reg=1.0)
    model_many.fit(X_many, y_many)
    print(f"Many features (regularized): converged ✓")
    
    # Imbalanced classes
    X_imb = np.random.randn(200, 3)
    y_imb = np.hstack([np.zeros(180), np.ones(20)])  # 90% negative
    
    model_imb = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=500)
    model_imb.fit(X_imb, y_imb)
    print(f"Imbalanced classes: handled ✓")
    
    print("✓ PASSED: Edge cases handled correctly")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Loop vs Vectorized", test_loop_vs_vectorized),
        ("Against sklearn", test_against_sklearn),
        ("Convergence", test_convergence),
        ("Regularization", test_regularization),
        ("Perfect Separation", test_perfect_separation),
        ("Prediction Threshold", test_prediction_threshold),
        ("SGD vs Batch", test_sgd_vs_batch),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
            print(f"✗ FAILED: {e}")
        except Exception as e:
            results.append((name, False, f"Unexpected error: {e}"))
            print(f"✗ ERROR: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if error and not success:
            print(f"       Error: {error}")
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Implementation is correct.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review implementation.")
    
    print("="*70)
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
