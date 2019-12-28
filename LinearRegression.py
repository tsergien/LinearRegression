#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from Regressor import Regressor
from scaling import Scaler

#  A program that calculates the precision of your algorithm. ????


class LinearRegression:
    '''Class for training data and graphing results'''
    def __init__(self, epochs=3000, l_rate=0.001) -> None:
        self.epochs: int = epochs
        self.l_rate: float = l_rate
        self.estimator = Regressor(0, 0)


    def graph(self, data: np.ndarray, dots: np.ndarray):
        x = np.linspace(data.min(0)[0]-1, data.max(0)[0]+1, 100)
        y = self.estimator.predict(x)
        plt.plot(x, y, 'k')
        plt.xlabel('mileage')
        plt.ylabel('estimated price')
        # plt.xlim(data.min(0)[0]-1, data.max(0)[0]+1)
        plt.scatter(dots[:,0], dots[:, 1])
        plt.title(f'After {self.epochs} iteration')
        plt.show()


    def loss_function(self, data: np.ndarray):
        m = data.shape[0]
        return sum([ (self.estimator.predict(data[j][0]) - data[j][1])**2 for j in range(m)])


    def train(self, df: pd.DataFrame, plot=True):
        np.random.seed(7171)

        scaled_data, mus, sigmas = Scaler.rescale(df)
        self.estimator.set_scaling_parameters(mus, sigmas)
        data = (np.matrix([scaled_data.km, scaled_data.price]).T).A
        m = data.shape[0]

        for epoch in range(self.epochs):
            tmp0 = self.l_rate * sum([ self.estimator.predict(data[j][0]) - data[j][1] for j in range(m)]) / m
            tmp1 = self.l_rate * sum([ (self.estimator.predict(data[j][0]) - data[j][1]) * data[j][0] for j in range(m)]) / m
            self.estimator.weights_update(tmp0, tmp1)

        if plot:
            self.graph(data, data)

            unscaled = np.reshape( np.asarray([df.km, self.estimator.predict(data[:,0]) * sigmas[1] + mus[1]]), (m, 2))
            self.graph(unscaled, (np.matrix([df.km, df.price]).T).A)
            
        
        print(f'Resulting loss function: {self.loss_function(data)}')

        pickle.dump(self.estimator, open('weights.sav', 'wb'))


        
        
