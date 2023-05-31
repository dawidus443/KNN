import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# pobranie danych
iris_data_train = pd.read_csv(r'iris_training_e.csv')
iris_data_validation = pd.read_csv(r'iris_validation_e.csv')

for column in iris_data_train.columns:
    if iris_data_train[column].dtypes != "float64":
        labels = {}
        unique_values = iris_data_train[column].unique()
        for i in range(len(unique_values)):
            labels[unique_values[i]] = i
        iris_data_train[column] = iris_data_train[column].map(labels)
#23-27
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

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
mlp_y_pred = mlp.predict(X_validation)
mlp_accuracy = accuracy_score(y_validation, mlp_y_pred)
print("Dokładność klasyfikacji: {:.2%}".format(mlp_accuracy), "\n")

# obliczenie macierzy pomyłek


confusion_matrix = ConfusionMatrix()
print('Iris-setosa: ')
confusion_matrix.measure(mlp_y_pred, y_validation, 0)
print('Iris-versicolor: ')
confusion_matrix.measure(mlp_y_pred, y_validation, 1)
print('Iris-virginica: ')
confusion_matrix.measure(mlp_y_pred, y_validation, 2)

# SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
svm_y_pred = svm.predict(X_validation)
svm_accuracy = accuracy_score(y_validation, svm_y_pred)
print("Dokładność klasyfikacji: {:.2%}".format(svm_accuracy), "\n")

print('Iris-setosa: ')
confusion_matrix.measure(svm_y_pred, y_validation, 0)
print('Iris-versicolor: ')
confusion_matrix.measure(svm_y_pred, y_validation, 1)
print('Iris-virginica: ')
confusion_matrix.measure(svm_y_pred, y_validation, 2)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_y_pred = nb.predict(X_validation)
nb_accuracy = accuracy_score(y_validation, nb_y_pred)
print("Dokładność klasyfikacji: {:.2%}".format(nb_accuracy), "\n")

print('Iris-setosa: ')
confusion_matrix.measure(nb_y_pred, y_validation, 0)
print('Iris-versicolor: ')
confusion_matrix.measure(nb_y_pred, y_validation, 1)
print('Iris-virginica: ')
confusion_matrix.measure(nb_y_pred, y_validation, 2)
