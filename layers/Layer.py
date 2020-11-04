import numpy as np
from functions.Activation import Activation

class Layer:
    """ Main Layer Class
        must be subclassed """
    def __init__(self, activation):
        self.activation = Activation(activation)
        self.z_cached = []
        self.a_prev = []
        self.w = [] #momentum
        self.b = [] #momentum
        self.vdw = [] #momentum
        self.vdb = [] #momentum
        self.sdw = [] #rmsprop
        self.sdb = [] #rmsprop

    def forward(self, input):
        raise NotImplementedError

    def update(self, dW, dB, learning_rate):
        self.w -= learning_rate*dW
        self.b -= learning_rate*dB

    def load(self, weights, biases):
        self.w = weights
        self.b = biases

    def save(self):
        return self.w, self.b