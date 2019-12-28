#!/usr/bin/env python3


class Regressor:
    def __init__(self, theta0=0, theta1=0) -> None:
        self.theta0 = theta0
        self.theta1 = theta1

    
    def predict(self, mileage) -> float:
        '''Based on the mileage calculates prize'''
        return mileage * self.theta1 + self.theta0


    def set_scaling_parameters(self, mus, sigmas):
        self.mu = mus[0]
        self.sigma = sigmas[0]
        self.mu2 = mus[1]
        self.sigma2 = sigmas[1]


    def get_scaling_parameters(self):
        return [self.mu, self.mu2], [self.sigma, self.sigma2]


    def get_weights(self):
        return self.theta0, self. theta1
    

    def weights_update(self, theta0: float, theta1: float) -> None:
        '''Sets parameters for predictions
        makes step in the antigradietds direction
        '''
        self.theta0 -= theta0
        self.theta1 -= theta1