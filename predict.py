#!/usr/bin/env python3

import pickle
from Predictor import Predict

if __name__ == "__main__":

    miles = int(input('Please, enter mileage: '))

    estimator = pickle.load(open('weights.sav', 'rb'))
    mus, sigmas = estimator.get_scaling_parameters()
    mu, mu2 = mus[0], mus[1]
    sigma, sigma2 = sigmas[0], sigmas[1]

    scaled_mileage = (miles - mu) / sigma
    print(f'Estimated price is: {estimator.predict(scaled_mileage) * sigma2 + mu2}')