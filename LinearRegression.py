#!usr/bin/env python3

import numpy as np
import pickle
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from Regressor import Regressor
from scaling import Scaler


class LinearRegression:
    '''Class for training data and graphing results'''
    def __init__(self, epochs=1000, l_rate=0.001) -> None:
        self.epochs: int = epochs
        self.l_rate: float = l_rate
        self.estimator = Regressor(0, 0)


    def graph(self, xg, yg, dots: np.ndarray, c='b', title='Graph'):
        '''Visualizing scattered dots and xg and yg as a line (prediction)'''
        plt.plot(xg, yg)
        plt.xlabel('mileage')
        plt.ylabel('estimated price')
        plt.xlim(xg.min()-1, xg.max()+1)
        plt.scatter(dots[:,0], dots[:, 1], marker='x')
        plt.title(title)
        plt.show()
        return


    def loss_function(self, data: np.ndarray):
        m = data.shape[0]
        return sum([ (self.estimator.predict(data[j][0]) - data[j][1])**2 for j in range(m)])


    def run_epoch(self, data: np.ndarray):
        m = data.shape[0]
        tmp0 = self.l_rate * sum([ self.estimator.predict(data[j][0]) - data[j][1] for j in range(m)]) / m
        tmp1 = self.l_rate * sum([ (self.estimator.predict(data[j][0]) - data[j][1]) * data[j][0] for j in range(m)]) / m
        self.estimator.weights_update(tmp0, tmp1)
        return


    def animated_training(self, data: np.ndarray, df: pd.DataFrame, mus, sigmas):
        fig, ax = plt.subplots(figsize=(16, 9), dpi=70)
        def animate(epoch: int):
            self.run_epoch(data)
            ax.clear()
            plt.title(f'epoch = {epoch}')
            ax.set_xlabel('km')
            ax.set_ylabel('price')
            ax.set_xlim(data.min(axis=0)[0]-1, data.max(axis=0)[0]+1)
            ax.set_ylim(-4, 4)
            x = np.linspace(start=data.min(axis=0)[0]-1, stop=data.max(axis=0)[0]+1, num=100)
            y = self.estimator.predict(x)
            line = plt.plot(x, y, label='prediction')
            plt.scatter(data[:,0], data[:, 1], label='raw data', marker='x')
            plt.legend()
            return line,
        ani = animation.FuncAnimation(fig, animate, frames=self.epochs, interval=10, blit=False)
        plt.show()
        for epoch in range(self.epochs):
            self.run_epoch(data)
        scaled_x = np.linspace(start=data.min(axis=0)[0]-1, stop=data.max(axis=0)[0]+1, num=100)
        self.graph(scaled_x, self.estimator.predict(scaled_x), data, 'k', f'Scaled data ({self.epochs})')
        x_lin = np.linspace(start=df.min(axis=0)[0]-1, stop=df.max(axis=0)[0]+1, num=100)
        y_lin = self.estimator.predict(scaled_x) * sigmas[1] + mus[1]
        self.graph(x_lin, y_lin, (np.matrix([df.km, df.price]).T).A, 'b', 'Resulting unscaled prediction')
        return


    def train(self, df: pd.DataFrame, plot=True):
        np.random.seed(7171)

        scaled_data, mus, sigmas = Scaler.rescale(df)
        self.estimator.set_scaling_parameters(mus, sigmas)
        data = (np.matrix([scaled_data.km, scaled_data.price]).T).A

        if not plot:
            for epoch in range(self.epochs):
                self.run_epoch(data)

        else: 
            self.animated_training(data, df, mus, sigmas)
        
        pickle.dump(self.estimator, open('weights.sav', 'wb'))
        print(f'Resulting loss function: {self.loss_function(data)}')
        return        
