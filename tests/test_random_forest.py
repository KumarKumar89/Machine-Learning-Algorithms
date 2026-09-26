"""Smoke tests for the from-scratch Random Forest implementation."""

import numpy as np
from sklearn.datasets import make_classification

from random_forest_src.random_forest import RandomForestClassifier


class TestRandomForestClassifier:
    def _data(self):
        X, y = make_classification(
            n_samples=200, n_features=6, n_informative=4,
            n_redundant=1, random_state=3,
        )
        return X.astype(float), y

    def test_beats_random_chance(self):
        X, y = self._data()
        rf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=42)
        rf.fit(X, y)
        acc = float(rf.score(X, y))
        assert acc > 0.75, f"Training accuracy too low: {acc}"

    def test_output_labels_match_input_classes(self):
        X, y = self._data()
        rf = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=0)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert set(np.unique(preds)).issubset(set(np.unique(y)))
