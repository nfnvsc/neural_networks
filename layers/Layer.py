import numpy as np
from functions.Activation import Activation

class Layer:
    """ Main Layer Class
        must be subclassed """
    def __init__(self, activation):
        self.activation = Activation(activation)
        self.z_cached = []
        self.a_prev = []

    def forward(self, input):
        raise NotImplementedError

    def update(self, dW, dB, learning_rate):
        self.weights -= learning_rate*dW
        self.biases -= learning_rate*dB

    def load(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def save(self):
        return self.weights, self.biases