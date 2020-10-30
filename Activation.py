import numpy as np

from Function import Function

def sigmoid(x):
    return 1 / (1 - np.exp(-x))
    
def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))



class Activation(Function):
    def __init__(self, func):
        if func == "sigmoid":
            self._function = sigmoid
            self._derivative = sigmoid_derivative
            return
        if func == "relu":
            pass
    
    def calculate(self, x):
        return self._function(x)
    
    def derivative(self, x):
        return self._derivative(x)