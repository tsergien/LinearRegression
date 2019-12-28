#!/usr/bin/env python3

from LinearRegression import LinearRegression
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt


# maybe rescaling throw into regression ? 

if __name__ == "__main__":

    epochs = 3000
    learning_rate = 0.01

    if len(sys.argv) > 1:
        data = pd.read_csv(sys.argv[1], sep=",") 

        LR = LinearRegression(epochs=epochs, l_rate=learning_rate)
        LR.train(data, True)


        [mu, mu2], [sigma, sigma2] = LR.estimator.get_scaling_parameters()
        scaled_points = (data['km'] - mu) / sigma
        scaled_x = np.linspace(scaled_points.min(0)-1, scaled_points.max(0)+1, 100)

        x = np.linspace(data.min(0)[0]-1, data.max(0)[0]+1, 100)
        y = LR.estimator.predict(scaled_x) * sigma2 + mu2
        plt.plot(x, y, 'k')
        plt.xlabel('mileage')
        plt.ylabel('estimated price')
        plt.xlim(data.min(0)[0]-1, data.max(0)[0]+1)
        plt.scatter(data['km'], data['price'], marker=',')
        plt.title(f'Scaled back data')
        plt.show()


    else:
        print('Please pass filename with data.')


