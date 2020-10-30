"""
My First Deep Neural Network from Scratch
"""

import numpy as np
from Layer import Layer

class Activation:
    def sigmoid(x):
        return 1 / (1 - np.exp(-x))
    
    def sigmoid_derivative(x):
        return Activation.sigmoid(x)*(1 - Activation.sigmoid(x))

class Loss:
    def crossEntropy(y, y_hat):
        return (-1/y.shape[1])*np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

    def crossEntropy_derivative(y, a):
        #return (-y/a) + (1-y)/(1-a)
        return np.sum((-y/a) + (1-y)/(1-a), axis=1, keepdims=True).T


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
        print(f"Loss: {loss}")
        return out

    def calculateGradients(self, input, labels):
        dA = Loss.crossEntropy_derivative(labels, input)
        #print(f"dA : {dA.shape}")
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
    inp_size = 5
    num_iterations = 100
    learning_rate = 0.01

    X = np.random.rand(inp_size, 10)
    Y = np.random.rand(10, 1)

    nn = NeuralNetwork(0.01)
    #self, input_size, hidden_units, activation
    dense1 = nn.addLayer(Layer(inp_size, 6, None))
    dense2 = nn.addLayer(Layer(6, 1, None))


    for _ in range(num_iterations):
        out = nn.feedForward(X, Y)
        dW, dB = nn.calculateGradients(out, Y)

        nn.gradientDescent(dW, dB)
