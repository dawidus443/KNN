import numpy as np


class OneVsAll:
    def __init__(self, estimator, n_classes):
        self.estimators = [estimator() for i in range(n_classes)]

    def fit(self, X, y):

        #Trenowanie klasyfikatorów binarnych dla każdej z klas.
        for i, estimator in enumerate(self.estimators):
            # Przygotowanie etykiet binarnych dla i-tej klasy.
            y_i = np.where(y == i, 1, 0)

            print (y_i)

            # Trenowanie klasyfikatora binarnego dla i-tej klasy.
            estimator.fit(X, y_i)

    def predict(self, X):

        # Wybór klasy z najwyższym wynikiem prawdopodobieństwa dla każdej próbki. //indeks

        print([estimator.predict_proba(X) for estimator in self.estimators])
        return np.argmax([estimator.predict_proba(X) for estimator in self.estimators], axis=0)


    '''
     0 1 1 
    .4 .3 .7
    .1 .2 .3
    
    '''