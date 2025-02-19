import numpy as np
import pandas as pd
import joblib
from RSS_min import RSS

# I built a seperate class for the Linear Regression Model, to evaluate it in a more felxible format

class LinearRegressionModel:

    def __init__(self):
        self.weights = []
        self.bias = 0

    def fit(self,X,y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        self.weights,self.bias = RSS.coeffecients(X,y)

    def predict(self,X):
        X = np.array(X, dtype=float)
        predictions = []
        for i in range(len(X)):
            sum = 0
            for x in range(len(self.weights)):
              sum += self.weights[x]*X[i][x]
            sum += self.bias
            predictions.append(sum)
        return predictions



