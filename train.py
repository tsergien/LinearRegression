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

class Trainer:
    def __init__(self, epochs=10_000, l_rate=0.01) -> None:
        self.epochs: int = epochs
        self.l_rate: float = l_rate

    def train(self, data: np.ndarray, plot=True):
        np.random.seed(7171)
        m = data.shape[0]
        estimator = Predict(np.random.rand(), np.random.rand()) # 0, 0 ? is it going to work
        for epoch in range(self.epochs):
            tmp0 = self.l_rate * sum([ estimator.predict(data[j][0]) - data[j][1] for j in range(m)]) / m
            tmp1 = self.l_rate * sum([ (estimator.predict(data[j][0]) - data[j][1]) * data[j][0] for j in range(m)]) / m
            estimator.set_parameters(tmp0, tmp1)
            print(f'Theta0 = {tmp0}, Theta1 = {tmp1}')

        if plot:
            x = np.linspace(data.min(0)[0], data.max(0)[0], 100)
            y = estimator.predict(x)
            plt.plot(x, y, color='green')
            plt.xlabel('mileage')
            plt.ylabel('estimated price')
            plt.scatter(data[:,0], data[:, 1])
            plt.show()
