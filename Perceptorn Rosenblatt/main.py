import pandas as pd
import numpy as np

from ConfusionMatrix import ConfusionMatrix
from Perceptron import Perceptron

# pobranie danych
banknote_data_train = pd.read_csv(r'data_banknote_authentication_training.csv')
banknote_data_validation = pd.read_csv(r'data_banknote_authentication_validation.csv')

X_train = banknote_data_train.drop(['outputs'], axis = 1)
y_train = banknote_data_train['outputs']

X_validation = banknote_data_validation.drop(['outputs'], axis = 1)
y_validation = banknote_data_validation['outputs']

# zamiana dataframe na tablice
X_train = np.array(X_train)
y_train = np.array(y_train)
X_validation = np.array(X_validation)
y_validation = np.array(y_validation)


# Tworzenie instancji klasy Perceptron
num_features = X_train.shape[1]
perceptron = Perceptron(num_features)

# Uczenie modelu na zestawie treningowym
perceptron.train(X_train, y_train)

# Testowanie na odpowiednim zestawie
accuracy = perceptron.test(X_validation, y_validation)

y_pred = perceptron.predict_all(X_validation)

print("Dokładność: {:.2%}".format(accuracy), "\n")

confusion_matrix = ConfusionMatrix()
print('Tablica pomyłek: ')
confusion_matrix.measure(y_pred, y_validation, 1)