import numpy as np
from knnscratch import KNN

class MultiOutput:
    def __init__(self, base_knn, k=3, distance_metric='euclidean', weights='uniform'):
        self.base_knn = base_knn
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights

    def fit(self, X, y):
        self.models = []
        n_outputs = y.shape[1]

        for i in range(n_outputs):
            knn = KNN(k=self.k, distance_metric=self.distance_metric, weights=self.weights)
            knn.fit(X, y[:, i])
            self.models.append(knn)
            print("Actual model",self.models)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.column_stack(predictions)
