import pandas as pd
import numpy as np

from Accuracy import Accuracy
from ConfusionMatrix import ConfusionMatrix
from OneVsAll import OneVsAll
from BinaryClassifier import BinaryClassifier
from One_vs_All.PerceptronClassifier import PerceptronClassifier

# pobranie danych
iris_data_train = pd.read_csv(r'iris_training_e.csv')
iris_data_validation = pd.read_csv(r'iris_validation_e.csv')

# zamiana nazw output na liczby
for column in iris_data_train.columns:
    if iris_data_train[column].dtypes != "float64":
        labels = {}
        unique_values = iris_data_train[column].unique()
        for i in range(len(unique_values)):
            labels[unique_values[i]] = i
        iris_data_train[column] = iris_data_train[column].map(labels)

for column in iris_data_validation.columns:
    if iris_data_validation[column].dtypes != "float64":
        labels = {}
        unique_values = iris_data_validation[column].unique()
        for i in range(len(unique_values)):
            labels[unique_values[i]] = i
        iris_data_validation[column] = iris_data_validation[column].map(labels)


# Wymieszanie danych treningowych
iris_data_train = iris_data_train.sample(frac=1).reset_index(drop=True)

# Wymieszanie danych walidacyjnych
iris_data_validation = iris_data_validation.sample(frac=1).reset_index(drop=True)

X_train = iris_data_train.drop(['outputs'], axis = 1)
y_train = iris_data_train['outputs']

X_validation = iris_data_validation.drop(['outputs'], axis = 1)
y_validation = iris_data_validation['outputs']

# zamiana dataframe na tablice
X_train = np.array(X_train)
y_train = np.array(y_train)
X_validation = np.array(X_validation)
y_validation = np.array(y_validation)

# Uczenie klasyfikatora OneVsAll i klasyfikacja próbek w zbiorze testowym
ova = OneVsAll(estimator=PerceptronClassifier, n_classes=3)
ova.fit(X_train, y_train)
y_pred = ova.predict(X_validation)

# obliczenie dokładności klasyfikatora
accuracy = Accuracy()
accuracy.calculate(y_validation, y_pred)
print('Dokładność klasyfikacji: {:.2f}%'.format(accuracy.print() * 100), '\n')

# print(y_validation , "\n")
# print(y_pred , "\n")

# obliczenie macierzy pomyłek
confusion_matrix = ConfusionMatrix()
print('Iris-setosa: ')
confusion_matrix.measure(y_pred, y_validation, 0)
print('Iris-versicolor: ')
confusion_matrix.measure(y_pred, y_validation, 1)
print('Iris-virginica: ')
confusion_matrix.measure(y_pred, y_validation, 2)