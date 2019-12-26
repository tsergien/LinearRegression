#!/usr/bin/env python3

import pickle
from Predictor import Predict

if __name__ == "__main__":

    miles = int(input('Please, enter mileage: '))
    

    estimator = pickle.load(open('weights.sav', 'rb'))
    print(f'Estimated price is: {estimator.predict(miles)}')