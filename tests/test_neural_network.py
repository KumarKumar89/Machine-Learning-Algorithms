"""Smoke tests for the from-scratch neural network (MLP) implementation."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from neural_network_src.neural_network import NeuralNetworkClassifier


class TestNeuralNetworkClassifier:
    def _data(self):
        X, y = make_classification(
            n_samples=200, n_features=4, n_informative=3,
            n_redundant=0, random_state=11,
        )
        X = StandardScaler().fit_transform(X)
        return X, y

    def test_learns_and_reduces_loss(self):
        X, y = self._data()
        nn = NeuralNetworkClassifier(
            layer_sizes=[4, 8, 2], learning_rate=0.1, random_state=42
        )
        nn.fit(X, y, epochs=600)
        assert nn.loss_history[-1] < nn.loss_history[0]
        acc = float(nn.score(X, y))
        assert acc > 0.75, f"Accuracy too low: {acc}"

    def test_proba_rows_sum_to_one(self):
        X, y = self._data()
        nn = NeuralNetworkClassifier(
            layer_sizes=[4, 6, 2], learning_rate=0.1, random_state=0
        )
        nn.fit(X, y, epochs=100)
        proba = nn.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)
