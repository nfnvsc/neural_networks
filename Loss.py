import numpy as np

from Function import Function
C = 1e-7

def cross_entropy(y, y_hat):
    m = y.shape[1]
    x = -(1/m)*np.sum(y*np.log(y_hat+C) + (1-y)*np.log(1-y_hat+C), axis=1, keepdims=True)
    return np.squeeze(x)

def cross_entropy_derivative(y, a):
    return -(np.divide(y, a) - np.divide(1-y, 1-a))

class Loss(Function):
    def __init__(self, func):
        if func == "cross_entropy":
            self._function = cross_entropy
            self._derivative = cross_entropy_derivative
    
    def calculate(self, y, a):
        return self._function(y, a)

    def derivative(self, y, a):
        return self._derivative(y, a)