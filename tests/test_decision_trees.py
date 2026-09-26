"""Smoke tests for the from-scratch Decision Tree implementation."""

import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split

from decision_trees_src.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class TestDecisionTreeClassifier:
    def test_iris_accuracy(self):
        iris = load_iris()
        X_tr, X_te, y_tr, y_te = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
        )
        clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
        clf.fit(X_tr, y_tr)
        acc = float(np.mean(clf.predict(X_te) == y_te))
        assert acc > 0.85, f"Iris accuracy too low: {acc}"

    def test_predict_proba_rows_sum_to_one(self):
        iris = load_iris()
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(iris.data, iris.target)
        proba = clf.predict_proba(iris.data[:20])
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(20), atol=1e-6)


class TestDecisionTreeRegressor:
    def test_regression_fit(self):
        X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=1)
        reg = DecisionTreeRegressor(max_depth=6)
        reg.fit(X, y)
        pred = reg.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.8, f"R^2 too low: {r2}"
