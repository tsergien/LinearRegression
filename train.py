#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
from Predictor import Predict

#  A program that calculates the precision of your algorithm.

# loss            : 1/2m sum (pred[i] - price[i])^2
# loss derivative : 1/m sum (pred[i] - price[i]) * x_i


class Trainer:
    def __init__(self, epochs=3000, l_rate=0.001) -> None:
        self.epochs: int = epochs
        self.l_rate: float = l_rate
        self.estimator = Predict(0, 0)


    def graph(self, data: np.ndarray):
        x = np.linspace(data.min(0)[0]-1, data.max(0)[0]+1, 100)
        y = self.estimator.predict(x)
        plt.plot(x, y, 'k')
        plt.xlabel('mileage')
        plt.ylabel('estimated price')
        plt.xlim(data.min(0)[0]-1, data.max(0)[0]+1)
        plt.scatter(data[:,0], data[:, 1])
        plt.title(f'After {self.epochs} iteration')
        plt.show()


    def train(self, data: np.ndarray, plot=True):
        np.random.seed(7171)
        m = data.shape[0]
        for epoch in range(self.epochs):
            tmp0 = self.l_rate * sum([ self.estimator.predict(data[j][0]) - data[j][1] for j in range(m)]) / m
            tmp1 = self.l_rate * sum([ (self.estimator.predict(data[j][0]) - data[j][1]) * data[j][0] for j in range(m)]) / m
            self.estimator.weights_update(tmp0, tmp1)

        print(f'Resulting loss function: {sum([ (self.estimator.predict(data[j][0]) - data[j][1])**2 for j in range(m)])}')

        pickle.dump(self.estimator, open('weights.sav', 'wb'))

        if plot:
            self.graph(data)

        
            

