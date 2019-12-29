#!/usr/bin/env python3

import pickle
from colors import color, red, blue
from Regressor import Regressor


if __name__ == "__main__":

    print(color(' You can enter car\'s mileage in km and get estimated price. To exit from program type \'exit\' :)', '#9ADDEC'))

    while 1:
        inp = str(input(color('\nPlease, enter mileage: ', '#9ADDEC')))
        if inp == "exit":
            exit()
        elif not inp.isnumeric():
            print(color('Please, enter positive number.', '#FF0000'))
        else:
            miles = int(inp)
            estimator = pickle.load(open('weights.sav', 'rb'))
            [mu, mu2], [sigma, sigma2] = estimator.get_scaling_parameters()

            scaled_mileage = (miles - mu) / sigma
            print(f'Estimated price is: ${int(estimator.predict(scaled_mileage) * sigma2 + mu2)}')
