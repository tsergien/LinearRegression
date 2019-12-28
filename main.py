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

    else:
        print('Please pass filename with data.')


