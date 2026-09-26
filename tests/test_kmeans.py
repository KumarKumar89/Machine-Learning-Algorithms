"""Smoke tests for the from-scratch K-Means implementations."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from kmeans_src.kmeans import KMeansScratch, KMeansVectorized


@pytest.fixture()
def blobs():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
    return X, y


@pytest.mark.parametrize(
    "cls", [KMeansScratch, KMeansVectorized], ids=["scratch", "vectorized"]
)
class TestKMeans:
    def test_recovers_clusters(self, cls, blobs):
        X, y = blobs
        km = cls(n_clusters=4, random_state=42)
        km.fit(X)
        labels = km.predict(X)
        assert adjusted_rand_score(y, labels) > 0.9

    def test_centroid_shape(self, cls, blobs):
        X, _ = blobs
        km = cls(n_clusters=4, random_state=42)
        km.fit(X)
        assert km.centroids.shape == (4, X.shape[1])

    def test_inertia_decreases_with_more_clusters(self, cls, blobs):
        X, _ = blobs
        inertia = []
        for k in (2, 4):
            km = cls(n_clusters=k, random_state=42)
            km.fit(X)
            inertia.append(km.inertia_)
        assert inertia[1] < inertia[0]
