import numpy as np

from functions.Function import Function

def cross_entropy(AL, Y):
    m = Y.shape[1]
    cost = -(1/m)*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    return np.squeeze(cost)

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