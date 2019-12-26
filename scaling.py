#!/usr/bin/env python3

import numpy as np
import pandas as pd
from statistics import mean, variance
import matplotlib.pyplot as plt


class Scaler():
    def __init__(self) -> None:
        return

    @staticmethod
    def rescale(data: pd.DataFrame) -> np.ndarray:
        means = data.mean(axis=0)
        variances = data.var(axis=0)

        scaled_data = data.copy(deep=True)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                scaled_data.iloc[row,col] = (data.iloc[row,col] - means[col]) / np.sqrt(variances[col])
        print(f'Not scaled: {data}')
        print(f'scaled: {scaled_data}')
        # plt.scatter(data.km, data.price, color='r')
        # plt.scatter(scaled_data.km, scaled_data.price, color='g')
        # plt.show()
        return scaled_data


