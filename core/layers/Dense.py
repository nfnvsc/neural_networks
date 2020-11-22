import numpy as np

from core.layers.Layer import Layer

class Dense(Layer):
    def __init__(self, input_size, neurons, activation, lambd = 1):
        super().__init__(activation)
        self._random_initialization(lambd, input_size, neurons)

    def _random_initialization(self, lambd, input_size, neurons):
        #if self.activation.func == "sigmoid":
        #    func = np.sqrt(lambd/neurons)
        #    self.weights = np.random.rand(neurons, input_size) * func
        #else:
        self.w = np.random.rand(neurons, input_size) * 0.01
        self.b = np.zeros((neurons,1))
        self.vdw = np.zeros((neurons, input_size))
        self.vdb = np.zeros((neurons, 1))
        self.sdw = np.zeros((neurons, input_size))
        self.sdb = np.zeros((neurons, 1))
    
    def forward(self, input):
        self.a_prev = input
        self.z_cached = np.dot(self.w, input) + self.b

        return self.activation.calculate(self.z_cached)
