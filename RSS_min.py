import numpy as np
import pandas as pd

class RSS:
    def coeffecients(X,y):
        weights = []
        x_means = np.mean(X,axis=0)
        y_mean = np.mean(y)
        col_data_y = y
        for col in range(X.shape[1]):
            col_data_x = X[:,col]
            col_data_x = col_data_x.reshape(-1, 1)
            col_data_y = col_data_y.reshape(-1, 1)

            if np.isnan(col_data_x).any() or np.isnan(col_data_y).any():
                continue

            variance = np.var(col_data_x)
            covariance = np.cov(col_data_x,col_data_y,rowvar=False)[0][1]
            if variance != 0:
                weight = covariance / variance
            else:
                weight = 0
            weights.append(weight)
        sum = 0
        for x in range(len(weights)):
            sum += weights[x]*x_means[x]
        bias = y_mean - sum

        return weights, bias