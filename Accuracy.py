import numpy as np

class Accuracy:
    def __init__(self):
        self.true_positives = 0
        self.total = 0

    def update(self, y_true, y_pred):
        self.true_positives += np.sum(y_true == y_pred)
        self.total += len(y_true)
        print(self.total, self.true_positives)

    def evaluate(self):
        return self.true_positives / self.total