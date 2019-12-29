#!/usr/bin/env python3

from LinearRegression import LinearRegression
import numpy as np
import pandas as pd
import sys


if __name__ == "__main__":

    epochs = 3000
    learning_rate = 0.01

    if len(sys.argv) > 1:
        try:
            data = pd.read_csv(sys.argv[1], sep=",") 
            LR = LinearRegression(epochs=epochs, l_rate=learning_rate)
            LR.train(data, True)
        except:
            print('Error opening or reading file.')

    else:
        print('Please pass filename with data.')
