import numpy as np

from functions.Function import Function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def rectified_linear(x):
    return np.maximum(0, x)

def rectified_linear_derivative(x):
    return (x > 0).astype(np.float)

class Activation(Function):
    def __init__(self, func):
        self.func = func
        if func == "sigmoid":
            self._function = sigmoid
            self._derivative = sigmoid_derivative

        if func == "relu":
            self._function = rectified_linear
            self._derivative = rectified_linear_derivative