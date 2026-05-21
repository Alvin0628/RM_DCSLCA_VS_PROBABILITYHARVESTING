import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class FeatureSubsetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, feature_indices):
        self.estimator       = estimator
        self.feature_indices = np.asarray(feature_indices)

    def fit(self, X, y, **kw):
        X = np.array(X)
        self.estimator.fit(X[:, self.feature_indices], y, **kw)
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict(self, X):
        X = np.array(X)
        return self.estimator.predict(X[:, self.feature_indices])

    def predict_proba(self, X):
        X = np.array(X)
        proba = self.estimator.predict_proba(X[:, self.feature_indices])
        if proba.shape[1] == 3:
            return proba
        full = np.zeros((len(X), 3))
        for i, c in enumerate(self.estimator.classes_):
            full[:, int(c)] = proba[:, i]
        return full