#!/usr/bin/env python3

from train import Trainer
import numpy as np
import pandas as pd
import sys

from scaling import Scaler

if __name__ == "__main__":
    trainer = Trainer()


    if len(sys.argv) > 1:
        data = pd.read_csv(sys.argv[1], sep=",") 
        scaled_data = Scaler.rescale(data)

        m = np.matrix([scaled_data.km, scaled_data.price]).T

        trainer.train(m.A)

    else:
        print('Please pass filename with data.')

