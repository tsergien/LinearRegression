#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from Predictor import Predict
# Plotting the data into a graph to see their repartition.
# 
# Plotting the line resulting from your linear regression into the same graph, to see
# the result of your hard work !
#
#  A program that calculates the precision of your algorithm.

# loss            : 1/2m sum (pred[i] - price[i])^2
# loss derivative : 1/m sum (pred[i] - price[i]) * x_i


class Trainer:
    def __init__(self, epochs=20000, l_rate=0.001) -> None:
        self.epochs: int = epochs
        self.l_rate: float = l_rate

    def train(self, data: np.ndarray, plot=True):
        np.random.seed(7171)
        m = data.shape[0]
        estimator = Predict(0, 0)
        for epoch in range(self.epochs):
            # print(f'loss function: {sum([ (estimator.predict(data[j][0]) - data[j][1])**2 for j in range(m)])}')
            # print(f'sum = {sum([ estimator.predict(data[j][0]) - data[j][1] for j in range(m)])}')
            # print(f'sum = {sum([ (estimator.predict(data[j][0]) - data[j][1])*data[j][0] for j in range(m)])}')
            # tmp0 = self.l_rate * sum([ estimator.predict(data[j][0]) - data[j][1] for j in range(m)]) / m
            # tmp1 = self.l_rate * sum([ (estimator.predict(data[j][0]) - data[j][1]) * data[j][0] for j in range(m)]) / m
            # estimator.weights_update(tmp0, tmp1)

            # online .
            for j in range(m):
                tmp0 = self.l_rate * estimator.predict(data[j][0]) - data[j][1]
                tmp1 = self.l_rate * (estimator.predict(data[j][0]) - data[j][1]) * data[j][0]
                estimator.weights_update(tmp0, tmp1)
            # print(f'tmp0= {tmp0}, tmp1 = {tmp1}')

        if plot:
            x = np.linspace(data.min(0)[0], data.max(0)[0], 100)
            y = estimator.predict(x)
            plt.plot(x, y, 'k')
            plt.xlabel('mileage')
            plt.ylabel('estimated price')
            plt.xlim(data.min(0)[0]-1, data.max(0)[0]+1)
            plt.scatter(data[:,0], data[:, 1])
            plt.title(f'After {self.epochs} iteration')
            plt.show()

