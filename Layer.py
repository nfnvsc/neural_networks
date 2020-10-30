import numpy as np
from Activation import Activation

class Layer:
    def __init__(self, input_size, neurons, activation, lambd = 1):
        self._random_initialization(lambd, input_size, neurons)
        self.activation = Activation(activation)
        self.z_cached = []

    def _random_initialization(self, lambd, input_size, neurons):
        func = np.sqrt(lambd/input_size)

        self.weights = np.random.rand(input_size, neurons) * func
        self.biases = np.zeros((neurons,1))
    
    def computeOutput(self, input):
        self.a_prev = input
        self.z_cached = np.dot(self.weights.T, input) + self.biases
        
        return Activation.calculate(self.z_cached)
    
    def computeGradient(self, dA, m):
        dZ = np.multiply(dA, Activation.derivative(self.z_cached))

        dW = 1/m * np.dot(dZ, self.a_prev.T)
        dB = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        new_dA = np.dot(self.weights, dZ)

        return dW, dB, new_dA
    
    def update(self, dW, dB, learning_rate):
        self.weights -= learning_rate*dW.T
        self.biases -= learning_rate*dB

