#!/usr/bin/env python3

from train import Trainer
import numpy as np
import pandas as pd
import sys
from scaling import Scaler
import matplotlib.pyplot as plt

# scatter before scaling
# scatter after
# rescale back?

if __name__ == "__main__":

    epochs = 3000
    learning_rate = 0.01

    if len(sys.argv) > 1:
        data = pd.read_csv(sys.argv[1], sep=",") 
        scaled_data, mus, sigmas = Scaler.rescale(data)

        m = np.matrix([scaled_data.km, scaled_data.price]).T

        trainer = Trainer(epochs=epochs, l_rate=learning_rate, mus=mus, sigmas=sigmas)
        trainer.train(m.A, True)


    else:
        print('Please pass filename with data.')

