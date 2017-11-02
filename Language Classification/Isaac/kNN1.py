import math
import pickle
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class KNN(object):
    def __init__(self, k):
        self.k = k
        self.X_train = np.asarray([])
        self.y_train = np.asarray([])
        self.X_test = np.asarray([])
        self.y_predict = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclideanDistance(self, a, b):
        distance = 0

        for x in range(a.shape[0] - 1):
            distance += (a[x] - b[x])**2
        return distance

    def _getNeighbors(self, instance):
        distances = []
        for i in range(self.X_test.shape[0]):
            dist = self._euclideanDistance(instance, self.X_train[i])
            distances.append((dist, i))
            distances = sorted(distances)

        neighbors = []
        for j in range(self.k):
            index = distances[j][1]
            neighbors.append(self.y_train[index])
        return neighbors

    def _getResponse(self, neighbors):
        return Counter(neighbors).most_common(1)[0][0]

    def accuracy(self, y_true):
        return accuracy_score(y_true, self.y_predict)

    def predict(self, X_test):
        self.X_test = X_test
        for x in range(self.X_test.shape[0]):
            neighbors = self._getNeighbors(self.X_test[x])
            result = self._getResponse(neighbors)
            self.y_predict.append(result)


def main():
    results = open("results.txt", "w")
    X = pickle.load(open('X.p', 'rb'))
    y = pickle.load(open('y.p', 'rb'))

    small_X = X[1:5000, :]
    small_y = y[1:5000]

    k_values = [3, 6, 9]
    dimensions = [20, 40, 60]
    for k_value in k_values:
        for dimension in dimensions:

            dim_reduce = PCA(dimension)
            reduced_x = dim_reduce.fit_transform(small_X.toarray())
            X_train, X_test, y_train, y_test = train_test_split(
                reduced_x, small_y)

            model = KNN(k_value)
            model.fit(X_train, y_train)
            model.predict(X_test)

            print(
                "Number of dimensions: " + str(dimension) + " k-value: " + str(k_value) +
                " accuracy: " + str(model.accuracy(y_test)))
    results.close()

main()
