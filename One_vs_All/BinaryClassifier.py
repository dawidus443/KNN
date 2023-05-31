import numpy as np


class BinaryClassifier:
    def __init__(self, lr=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # inicjalizacja wag i biasu
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # trenowanie modelu
        for _ in range(self.n_iterations):
            # obliczenie wyników dla każdej próbki
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # obliczenie gradientów i aktualizacja wag i biasu
            dw = np.dot(X.T, (y_pred - y))
            db = np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        # obliczenie wyników dla każdej próbki
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)

        # zwrócenie prawdopodobieństw klasyfikacji dla klasy pozytywnej
        return y_pred

    def _sigmoid(self, z):
        # funkcja sigmoidalna
        return 1 / (1 + np.exp(-z))










    # def predict(self, X):
    #     # obliczenie wyników dla każdej próbki i przypisanie etykiet
    #     linear_model = np.dot(X, self.weights) + self.bias
    #     y_pred = self._sigmoid(linear_model)
    #     y_pred_labels = (y_pred > 0.5).astype(int)
    #
    #     return y_pred_labels

