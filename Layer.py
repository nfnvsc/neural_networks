import numpy as np
from Activation import Activation

class Layer:
    def __init__(self, input_size, neurons, activation, lambd = 1):
        self._random_initialization(lambd, input_size, neurons)
        self.activation = Activation(activation)
        self.z_cached = []
        self.a_cached = []

    def _random_initialization(self, lambd, input_size, neurons):
        func = np.sqrt(lambd/input_size)

        self.weights = np.random.rand(neurons, input_size) * func
        self.biases = np.zeros((neurons,1))
    
    def computeOutput(self, input):
        self.a_prev = input

        self.z_cached = np.dot(self.weights, input) + self.biases
        
        return self.activation.calculate(self.z_cached)
    
    def computeGradient(self, dA, m):
        dZ = dA * self.activation.derivative(self.z_cached)

        dW = 1/m * np.dot(dZ, self.a_prev.T)
        dB = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        dA_prev = np.dot(self.weights.T, dZ)

        return dW, dB, dA_prev
    
    def update(self, dW, dB, learning_rate):
        #print("---dW---")
        #print(dW)
        #print("---weights-before---")
        #print(self.weights)
        self.weights -= learning_rate*dW
        #print("---weights-after---")
        #print(self.weights)
        self.biases -= learning_rate*dB

