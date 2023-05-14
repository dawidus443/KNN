import numpy as np


class OneVsAll:
    def __init__(self, estimator, n_classes):
        self.estimators = [estimator() for i in range(n_classes)]

    def fit(self, X, y):
        """
        Trenowanie klasyfikatorów binarnych dla każdej z klas.
        """
        for i, estimator in enumerate(self.estimators):
            # Przygotowanie etykiet binarnych dla i-tej klasy.
            y_i = np.where(y == i, 1, 0)

            # Trenowanie klasyfikatora binarnego dla i-tej klasy.
            estimator.fit(X, y_i)

    def predict(self, X):
        """
        Klasyfikacja próbek za pomocą klasyfikatorów binarnych.
        """
        # Wybór klasy z najwyższym wynikiem prawdopodobieństwa dla każdej próbki.
        return np.argmax([estimator.predict_proba(X)[:, 1] for estimator in self.estimators], axis=0)


