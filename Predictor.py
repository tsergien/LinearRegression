#!/usr/bin/env python3


class Predict:
    def __init__(self, theta0=0, theta1=0) -> None:
        self.theta0 = theta0
        self.theta1 = theta1

    
    def predict(self, mileage) -> float:
        '''Based on the mileage calculates prize'''
        return mileage * self.theta1 + self.theta0


    def set_parameters(self, theta0: float, theta1: float) -> None:
        '''Sets parameters for predictions'''
        self.theta0 = theta0
        self.theta1 = theta1

    def weights_update(self, theta0: float, theta1: float) -> None:
        '''Sets parameters for predictions
        makes step in the antigradietds direction
        '''
        self.theta0 -= theta0
        self.theta1 -= theta1