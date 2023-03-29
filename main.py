import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from KNN import KNN
from Accuracy import Accuracy
from ConfusionMatrix import ConfusionMatrix

# pobranie danych
iris_data_train = pd.read_csv(r'iris_training_e.csv')
iris_data_validation = pd.read_csv(r'iris_validation_e.csv')

# zamiana nazw output na liczby
for column in iris_data_train.columns:
    if iris_data_train[column].dtype == np.float64:
        continue
iris_data_train[column] = LabelEncoder().fit_transform(iris_data_train[column])

for column in iris_data_validation.columns:
    if iris_data_validation[column].dtype == np.float64:
        continue
iris_data_validation[column] = LabelEncoder().fit_transform(iris_data_validation[column])

# wymieszanie danych (?)
np.random.shuffle(iris_data_train.values)
np.random.shuffle(iris_data_validation.values)

X_train = iris_data_train.drop(['outputs'], axis = 1)
y_train = iris_data_train['outputs']

X_validation = iris_data_validation.drop(['outputs'], axis = 1)
y_validation = iris_data_validation['outputs']

# zamiana dataframe na tablice
X_train = np.array(X_train)
y_train = np.array(y_train)
X_validation = np.array(X_validation)
y_validation = np.array(y_validation)

print(X_train, "\n", X_validation, "\n", y_train, "\n", y_validation)

# inicjalizacja i dopasowanie klasyfikatora
knn = KNN(k=3)
knn.fit(X_train, y_train)

# predykcja etykiet dla danych testowych
y_pred = knn.predict(X_validation)

# obliczenie dokładności klasyfikatora
accuracy = Accuracy()
accuracy.update(y_validation, y_pred)
print('Dokładność klasyfikacji: {:.2f}%'.format(accuracy.evaluate() * 100), '\n')

# obliczenie macierzy pomyłek
confusion_matrix = ConfusionMatrix()
print('Iris-setosa: ')
confusion_matrix.measure(y_pred, y_validation, 0)
print('Iris-versicolor: ')
confusion_matrix.measure(y_pred, y_validation, 1)
print('Iris-virginica: ')
confusion_matrix.measure(y_pred, y_validation, 2)
