import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.weights == 'distance':
            weights = [1.0 / (distances[i] + 1e-5) for i in k_indices]  # Avoid division by zero
            weighted_labels = Counter()
            for label, weight in zip(k_nearest_labels, weights):
                weighted_labels[label] += weight
            most_common = weighted_labels.most_common(1)
            return most_common[0][0]
        else:
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]

    def _compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
