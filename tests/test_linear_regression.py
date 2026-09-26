"""Smoke tests for the from-scratch Linear Regression implementation."""

import numpy as np
import pytest

from linear_regression_src.vectorized_implementation import LinearRegressionVectorized


def _make_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, size=(n, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w + 3.0 + rng.normal(0, 0.1, size=n)
    return X, y


class TestLinearRegressionVectorized:
    def test_fit_reduces_loss(self):
        X, y = _make_data()
        model = LinearRegressionVectorized(learning_rate=0.05, n_iterations=800)
        model.fit(X, y)
        assert len(model.loss_history) > 1
        assert model.loss_history[-1] < model.loss_history[0]

    def test_recovers_signal(self):
        X, y = _make_data()
        model = LinearRegressionVectorized(learning_rate=0.05, n_iterations=1500)
        model.fit(X, y)
        pred = model.predict(X)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        assert rmse < 0.5, f"RMSE too high: {rmse}"

    def test_matches_sklearn(self):
        from sklearn.linear_model import LinearRegression

        X, y = _make_data()
        model = LinearRegressionVectorized(learning_rate=0.05, n_iterations=3000)
        model.fit(X, y)
        ours = model.predict(X[:10])
        ref = LinearRegression().fit(X, y).predict(X[:10])
        np.testing.assert_allclose(ours, ref, rtol=0.05, atol=0.2)

    def test_score_is_r_squared(self):
        X, y = _make_data()
        model = LinearRegressionVectorized(learning_rate=0.05, n_iterations=2000)
        model.fit(X, y)
        r2 = model.score(X, y)
        assert 0.9 < r2 <= 1.0
