import numpy as np


class PerceptronClassifier:
    def __init__(self, lr=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            for i in range(len(X)):
                linear_model = np.dot(X[i], self.weights) + self.bias
                y_pred = self._activation(linear_model)

                self.weights += self.lr * (y[i] - y_pred) * X[i]
                self.bias += self.lr * (y[i] - y_pred)

    def predict_proba(self, X):
        y_pred = np.array([(np.dot(x, self.weights) + self.bias) for x in X])
        return y_pred

    def _activation(self, z):
        return 1 if z >= 0 else 0