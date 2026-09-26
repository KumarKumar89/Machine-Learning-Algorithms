"""Smoke tests for the from-scratch Logistic Regression implementation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from logistic_regression_src.vectorized_implementation import LogisticRegressionVectorized, sigmoid


class TestSigmoid:
    def test_output_range(self):
        z = np.linspace(-50, 50, 101)
        s = sigmoid(z)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_sigmoid_at_zero(self):
        assert pytest.approx(float(np.ravel(sigmoid(np.array([0.0])))[0]), abs=1e-6) == 0.5


class TestLogisticRegressionVectorized:
    def _data(self):
        X, y = make_classification(
            n_samples=300, n_features=5, n_informative=3,
            n_redundant=0, random_state=7,
        )
        return X.astype(float), y.astype(float)

    def test_learns_separable_data(self):
        X, y = self._data()
        model = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(np.mean(preds == y))
        assert accuracy > 0.85, f"Accuracy too low: {accuracy}"

    def test_probabilities_are_valid(self):
        X, y = self._data()
        model = LogisticRegressionVectorized(learning_rate=0.1, n_iterations=500)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)
