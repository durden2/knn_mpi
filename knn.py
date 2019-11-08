from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np

class knn:
    def __init__(self):
        self.load_data()
        self.split_data()

    def load_data(self):
        self._mnist = datasets.load_digits()

    def split_data(self):
        (self._trainData, self._testData, self._trainLabels, self._testLabels) = \
            train_test_split(np.array(self._mnist.data), self._mnist.target, test_size=0.25, random_state=42)

    def train(self, number_of_neighbors):
        model = KNeighborsClassifier(n_neighbors=number_of_neighbors)
        model.fit(self._trainData, self._trainLabels)

        score = model.score(self._testData, self._testLabels)
        return score