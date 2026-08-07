"""
Unit tests for Linear Regression implementations.

Tests verify:
1. Correctness against known solutions
2. Numerical precision
3. Edge cases
4. Consistency between implementations
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from loop_implementation import LinearRegressionLoop, generate_sample_data
from vectorized_implementation import LinearRegressionVectorized


def test_perfect_linear_relationship():
    """Test with noise-free linear data - should achieve R² = 1.0"""
    print("\n" + "="*60)
    print("TEST 1: Perfect Linear Relationship")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_w = np.array([2.0, -1.5, 3.0])
    true_b = 5.0
    y = X @ true_w + true_b  # No noise
    
    # Test Normal Equation
    model = LinearRegressionVectorized()
    model.fit_normal_equation(X, y)
    
    r2 = model.score(X, y)
    weight_error = np.max(np.abs(model.coef_ - true_w))
    bias_error = abs(model.intercept_ - true_b)
    
    print(f"R² Score: {r2:.10f}")
    print(f"Max weight error: {weight_error:.2e}")
    print(f"Bias error: {bias_error:.2e}")
    
    assert r2 > 0.9999, f"R² should be ~1.0, got {r2}"
    assert weight_error < 1e-10, f"Weights should match exactly"
    assert bias_error < 1e-10, f"Bias should match exactly"
    
    print("✓ PASSED: Perfect recovery of parameters\n")


def test_with_noise():
    """Test with noisy data - should recover parameters approximately"""
    print("="*60)
    print("TEST 2: Noisy Linear Relationship")
    print("="*60)
    
    X, y = generate_sample_data(
        n_samples=500,
        n_features=5,
        true_weights=np.array([1.0, -2.0, 0.5, 3.0, -1.5]),
        true_bias=2.0,
        noise_std=0.1,
        random_seed=42
    )
    
    # Test Normal Equation
    model_ne = LinearRegressionVectorized()
    model_ne.fit_normal_equation(X, y)
    
    # Test Gradient Descent
    model_gd = LinearRegressionVectorized(learning_rate=0.01, n_iterations=2000)
    model_gd.fit(X, y)
    
    # Test Loop
    model_loop = LinearRegressionLoop(learning_rate=0.01, n_iterations=2000)
    model_loop.fit(X, y)
    
    r2_ne = model_ne.score(X, y)
    r2_gd = model_gd.score(X, y)
    r2_loop = model_loop.score(X, y)
    
    print(f"Normal Equation R²: {r2_ne:.4f}")
    print(f"Gradient Descent R²: {r2_gd:.4f}")
    print(f"Loop Implementation R²: {r2_loop:.4f}")
    
    # All should achieve similar R²
    assert abs(r2_ne - r2_gd) < 0.01, "NE and GD should have similar R²"
    assert abs(r2_ne - r2_loop) < 0.01, "NE and Loop should have similar R²"
    assert r2_ne > 0.95, f"R² should be high (>0.95), got {r2_ne}"
    
    print("✓ PASSED: All implementations achieve similar performance\n")


def test_consistency_with_sklearn():
    """Verify our implementation matches scikit-learn"""
    print("="*60)
    print("TEST 3: Consistency with scikit-learn")
    print("="*60)
    
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("⚠️  scikit-learn not available, skipping test\n")
        return
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randn(200)
    
    # Our implementation
    model_ours = LinearRegressionVectorized()
    model_ours.fit_normal_equation(X, y)
    
    # sklearn
    model_sklearn = LinearRegression(fit_intercept=True)
    model_sklearn.fit(X, y)
    
    # Compare
    weight_diff = np.max(np.abs(model_ours.coef_ - model_sklearn.coef_))
    bias_diff = abs(model_ours.intercept_ - model_sklearn.intercept_)
    r2_diff = abs(model_ours.score(X, y) - model_sklearn.score(X, y))
    
    print(f"Max weight difference: {weight_diff:.2e}")
    print(f"Bias difference: {bias_diff:.2e}")
    print(f"R² difference: {r2_diff:.2e}")
    
    assert weight_diff < 1e-10, f"Weights should match sklearn"
    assert bias_diff < 1e-10, f"Bias should match sklearn"
    assert r2_diff < 1e-10, f"R² should match sklearn"
    
    print("✓ PASSED: Matches scikit-learn within floating point precision\n")


def test_single_feature():
    """Test simple 1D case"""
    print("="*60)
    print("TEST 4: Single Feature (1D)")
    print("="*60)
    
    # Simple: y = 3x + 2
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([5, 8, 11, 14, 17])  # Exactly 3x + 2
    
    model = LinearRegressionVectorized()
    model.fit_normal_equation(X, y)
    
    print(f"Learned weight: {model.coef_[0]:.6f} (expected: 3.0)")
    print(f"Learned bias: {model.intercept_:.6f} (expected: 2.0)")
    print(f"R²: {model.score(X, y):.6f}")
    
    assert abs(model.coef_[0] - 3.0) < 1e-10
    assert abs(model.intercept_ - 2.0) < 1e-10
    assert model.score(X, y) > 0.9999
    
    print("✓ PASSED: Perfect recovery in 1D case\n")


def test_no_intercept():
    """Test fitting without intercept (through origin)"""
    print("="*60)
    print("TEST 5: No Intercept (Through Origin)")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_w = np.array([2.0, 3.0])
    y = X @ true_w  # No bias term
    
    model = LinearRegressionVectorized(fit_intercept=False)
    model.fit_normal_equation(X, y)
    
    print(f"Learned weights: {model.coef_}")
    print(f"True weights: {true_w}")
    print(f"Weight error: {np.max(np.abs(model.coef_ - true_w)):.2e}")
    
    assert model.intercept_ == 0.0, "Intercept should be 0 when fit_intercept=False"
    assert np.max(np.abs(model.coef_ - true_w)) < 1e-10
    
    print("✓ PASSED: Correctly fits through origin\n")


def test_multicollinearity():
    """Test behavior with correlated features"""
    print("="*60)
    print("TEST 6: Multicollinearity Warning")
    print("="*60)
    
    np.random.seed(42)
    n = 100
    
    # Create highly correlated features
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.01  # Almost perfectly correlated
    X = np.column_stack([x1, x2])
    
    y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.1
    
    model = LinearRegressionVectorized()
    model.fit_normal_equation(X, y)
    
    # With multicollinearity, individual weights may be unstable
    # but predictions should still be good
    r2 = model.score(X, y)
    print(f"R² Score: {r2:.4f}")
    print(f"Coefficients: {model.coef_}")
    
    # R² should still be high even if individual coefficients are unstable
    assert r2 > 0.95, "Predictions should still be accurate"
    
    print("✓ PASSED: Handles multicollinearity (predictions remain accurate)\n")


def test_large_dataset():
    """Test scalability with large dataset"""
    print("="*60)
    print("TEST 7: Large Dataset Scalability")
    print("="*60)
    
    import time
    
    np.random.seed(42)
    m, n = 10000, 50
    X = np.random.randn(m, n)
    y = np.random.randn(m)
    
    # Time Normal Equation
    start = time.time()
    model_ne = LinearRegressionVectorized()
    model_ne.fit_normal_equation(X, y)
    time_ne = time.time() - start
    
    # Time Gradient Descent (fewer iterations for speed)
    start = time.time()
    model_gd = LinearRegressionVectorized(learning_rate=0.01, n_iterations=100)
    model_gd.fit(X, y)
    time_gd = time.time() - start
    
    print(f"Normal Equation time: {time_ne:.3f}s")
    print(f"Gradient Descent time: {time_gd:.3f}s")
    print(f"Normal Equation R²: {model_ne.score(X, y):.4f}")
    print(f"GD R²: {model_gd.score(X, y):.4f}")
    
    # Both should complete in reasonable time
    assert time_ne < 10.0, f"NE should complete in <10s, took {time_ne:.1f}s"
    assert time_gd < 10.0, f"GD should complete in <10s, took {time_gd:.1f}s"
    
    print("✓ PASSED: Scales to large datasets\n")


def test_predictions_shape():
    """Test that predictions have correct shape"""
    print("="*60)
    print("TEST 8: Prediction Shape Consistency")
    print("="*60)
    
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    
    model = LinearRegressionVectorized()
    model.fit_normal_equation(X_train, y_train)
    
    # Test single sample
    X_single = np.random.randn(1, 5)
    pred_single = model.predict(X_single)
    assert pred_single.shape == (1,), f"Single prediction shape wrong: {pred_single.shape}"
    
    # Test batch
    X_batch = np.random.randn(50, 5)
    pred_batch = model.predict(X_batch)
    assert pred_batch.shape == (50,), f"Batch prediction shape wrong: {pred_batch.shape}"
    
    print("✓ PASSED: Predictions have correct shapes\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("LINEAR REGRESSION - UNIT TEST SUITE")
    print("="*60)
    
    tests = [
        test_perfect_linear_relationship,
        test_with_noise,
        test_consistency_with_sklearn,
        test_single_feature,
        test_no_intercept,
        test_multicollinearity,
        test_large_dataset,
        test_predictions_shape,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            failed += 1
    
    # Summary
    print("="*60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Implementation is correct.\n")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review implementation.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
