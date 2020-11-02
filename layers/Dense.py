import numpy as np

from Activation import Activation
from layers.Layer import Layer

class Dense(Layer):
    def __init__(self, input_size, neurons, activation, lambd = 1):
        super().__init__(activation)
        self._random_initialization(lambd, input_size, neurons)

    def _random_initialization(self, lambd, input_size, neurons):
        #if self.activation.func == "sigmoid":
        #    func = np.sqrt(lambd/neurons)
        #    self.weights = np.random.rand(neurons, input_size) * func
        #else:
        self.weights = np.random.rand(neurons, input_size) * 0.01
        self.biases = np.zeros((neurons,1))
    
    def forward(self, input):
        self.a_prev = input
        self.z_cached = np.dot(self.weights, input) + self.biases
        return self.activation.calculate(self.z_cached)
    
    def backward(self, dA):
        dZ = dA * self.activation.derivative(self.z_cached)
        m = self.a_prev.shape[1]

        dW = 1/m * np.dot(dZ, self.a_prev.T)
        dB = 1/m * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(self.weights.T, dZ)

        return dW, dB, dA_prev