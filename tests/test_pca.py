"""Smoke tests for the from-scratch PCA implementation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA as SklearnPCA

from pca_src.pca import PCAScratch


@pytest.fixture()
def data():
    X, _ = make_classification(
        n_samples=200, n_features=5, n_informative=3, n_redundant=0, random_state=5
    )
    return X.astype(float)


class TestPCAScratch:
    def test_transform_shape(self, data):
        pca = PCAScratch(n_components=2)
        Z = pca.fit_transform(data)
        assert Z.shape == (data.shape[0], 2)

    def test_variance_matches_sklearn(self, data):
        ours = PCAScratch(n_components=3).fit(data)
        ref = SklearnPCA(n_components=3).fit(data)
        np.testing.assert_allclose(
            ours.explained_variance_ratio_, ref.explained_variance_ratio_, atol=1e-6
        )

    def test_components_are_orthonormal(self, data):
        pca = PCAScratch(n_components=3).fit(data)
        W = pca.components_[:3] if pca.components_.shape[0] >= 3 else pca.components_
        gram = W @ W.T
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-6)

    def test_cumulative_variance_monotonic(self, data):
        pca = PCAScratch(n_components=4).fit(data)
        cum = np.asarray(pca.cumulative_explained_variance_ratio_)
        assert np.all(np.diff(cum) >= -1e-12)
        assert cum[-1] <= 1.0 + 1e-9
