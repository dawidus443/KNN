import numpy as np

class Accuracy:
    def __init__(self):
        self.true_y = 0
        self.total = 0

    def calculate(self, y_true, y_pred):
        self.true_y += np.sum(y_true == y_pred)
        self.total += len(y_true)

    def print(self):
        return self.true_y / self.total