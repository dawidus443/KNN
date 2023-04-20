import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.random.rand(num_features + 1) * 2 - 1  # Inicjalizacja losowych wektor wag w zakresie (-1, 1)


    def predict(self, x_valid):
        # il sk
        dot_product = np.dot(x_valid, self.weights[1:]) + self.weights[0]

        # wynik funkcji aktywacji
        if dot_product > 0:
            return 1
        else:
            return 0

    def train(self, x_train, y_train):
        for epoch in range(self.num_epochs):
            for x, y in zip(x_train, y_train):
                # Oblicz przewidywany wynik
                y_predicted = self.predict(x)

                # poprawianie wag
                if y_predicted != y:
                    error = y - y_predicted
                    self.weights[1:] += self.learning_rate * error * x
                    self.weights[0] += self.learning_rate * error

    def test(self, x_valid, y_valid):
        num_correct = 0
        for x, y in zip(x_valid, y_valid):
            y_predicted = self.predict(x)
            if y_predicted == y:
                num_correct += 1
        accuracy = num_correct / len(y_valid)
        return accuracy

    def predict_all(self, x_valid):
        y_pred = []
        for xi in x_valid:
            y_pred.append(self.predict(xi))
        return np.array(y_pred)

