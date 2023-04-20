import numpy as np


class ConfusionMatrix:

    def measure(self, y_pred, y_validation, klasa):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_validation)):
            if y_pred[i] == y_validation[i] == klasa:
                TP += 1
            elif y_pred[i] == klasa and y_validation[i] != y_pred[i]:
                FP += 1
            elif y_validation[i] == klasa and y_validation[i] != y_pred[i]:
                FN += 1
            else:
                TN += 1
        print("TP:", TP, " FP:", FP, "\nTN:", TN, " FN:", FN,)
        print("Accuracy: ", ((TP + TN) / (TP + TN + FP + FN)), " , precision: ",
              (TP / (TP + FP)), ", recall: ",
              (TP / (TP + FN)), '\n')
