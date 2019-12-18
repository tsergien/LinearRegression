#!/usr/bin/env python3

from train import Trainer
import numpy as np
import pandas as pd
import sys

if __name__ == "__main__":
    trainer = Trainer(10)

    # pass file name as parameter later
    data = pd.read_csv(sys.argv[1], sep=",") 

    m = np.matrix([data.km, data.price]).T
    print(f'matrix {m.shape}')

    trainer.train(m.A)