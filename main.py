import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from KNN import KNN
from Accuracy import Accuracy
from ConfusionMatrix import ConfusionMatrix

# normalizacja, skalowanie, standaryzacja


# df = pd.read_csv(r'C:\Users\dawid\Downloads\breast-cancer.csv')
# np.random.shuffle(df.values)
# print(df.values)

#
# iris_data_test = pd.read_csv(r'iris_test_e.csv')
# np.random.shuffle(iris_data_test.values)
#
# # print(iris_data_train['outputs'].unique(), iris_data_train.isnull().values.any(), iris_data_train['outputs'].value_counts())
#
# for column in iris_data_train.columns:
#     if iris_data_train[column].dtype == np.float64:
#         continue
# iris_data_train[column] = LabelEncoder().fit_transform(iris_data_train[column])
#
# for column in iris_data_test.columns:
#     if iris_data_test[column].dtype == np.float64:
#         continue
# iris_data_train[column] = LabelEncoder().fit_transform(iris_data_train[column])
#
# #print(iris_data_train.dtypes)
#
# X_train = iris_data_train.drop(['outputs'], axis = 1)
# y_train = iris_data_train['outputs']
#
# X_test = iris_data_train.drop(['outputs'], axis = 1)
# y_test = iris_data_train['outputs']
#
#
#
# k_range = list(range(1, 12))
# acc = []
#
# for i in k_range:
#     knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     acc.append(metrics.accuracy_score(y_test, y_pred))
#
# print(acc)

# iris = load_iris()
# X = iris.data
# y = iris.target

#############################################################################################################################################


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

# podział danych na zbiory treningowe i testowe
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train, "\n", X_validation, "\n", y_train, "\n", y_validation)

# inicjalizacja i dopasowanie klasyfikatora
knn = KNN(k=3)
knn.fit(X_train, y_train)

# predykcja etykiet dla danych testowych
y_pred = knn.predict(X_validation)

# obliczenie dokładności klasyfikatora
accuracy = Accuracy()
accuracy.update(y_validation, y_pred)
print('Dokładność klasyfikacji: {:.2f}%'.format(accuracy.evaluate() * 100))

# obliczenie macierzy pomyłek
confusion_matrix = ConfusionMatrix()
confusion_matrix.measure(y_pred, y_validation, 0)
confusion_matrix.measure(y_pred, y_validation, 1)
confusion_matrix.measure(y_pred, y_validation, 2)
