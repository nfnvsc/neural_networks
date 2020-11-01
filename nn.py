"""
My First Deep Neural Network from Scratch
"""

import numpy as np
from Layer import Layer

C = 1e-7

class Activation:
    def sigmoid(x):
        return 1 / (1 - np.exp(-x))
    
    def sigmoid_derivative(x):
        return Activation.sigmoid(x)*(1 - Activation.sigmoid(x))

class Loss:
    def crossEntropy(y, y_hat):
        m = y.shape[1]
        x = (-1/m)*np.sum(y*np.log(y_hat + C) + (1-y)*np.log(1-y_hat + C), axis=1, keepdims=True)
        return np.squeeze(x)

    def crossEntropy_derivative(y, a):
        x = - np.divide(y, a) - np.divide(1-y, 1-a)
        return x

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.layers = []
        self.learning_rate = learning_rate

    def addLayer(self, layer):
        self.layers.append(layer)

    def computeLoss(self, y, y_hat):
        return Loss.crossEntropy(y, y_hat)
    
    def feedForward(self, features, labels):
        out = features
        for l in self.layers:
            out = l.computeOutput(out)

        loss = self.computeLoss(out, labels)
        print(f"Loss: {np.sum(loss)}")
        return out

    def calculateGradients(self, input, labels):
        dA = Loss.crossEntropy_derivative(labels, input).T
        dWs = []
        dBs = []
        for l in range(len(self.layers)-1, -1, -1):
            dW, dB, dA = self.layers[l].computeGradient(dA, input.shape[1])
            dWs.append(dW)
            dBs.append(dB)

        return dWs, dBs        

    def gradientDescent(self, dWs, dBs):
        for i in range(len(self.layers)):
            dW = dWs[i]
            dB = dBs[i]
            self.layers[len(self.layers) - 1 - i].update(dW, dB, self.learning_rate)



if __name__ == "__main__":
    inp_size = 10
    num_iterations = 100
    learning_rate = 0.1

    X = np.random.rand(inp_size, 10)
    Y = np.random.rand(10, 1)

    nn = NeuralNetwork(learning_rate)
    #self, input_size, hidden_units, activation
    dense1 = nn.addLayer(Layer(inp_size, 10, "sigmoid"))
    dense2 = nn.addLayer(Layer(10, 1, "sigmoid"))


    for _ in range(num_iterations):
        out = nn.feedForward(X, Y)
        dW, dB = nn.calculateGradients(out, Y)
        nn.gradientDescent(dW, dB)
