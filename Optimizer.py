from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Interface for optimizers
    """
    @abstractmethod
    def backward(self, layers: list):
        pass

    @abstractmethod
    def update_parameters(self, layers: list, dW, dB):
        pass


class GradientDescent(Optimizer):
    def backward(self, loss_deriv, inp, labels, layers: list):
        dWs = []
        dBs = []
        dA = loss_deriv
        for layer in reversed(layers):
            dZ = dA * layer.activation.derivative(layer.z_cached)
            m = layer.a_prev.shape[1]

            dW = (1.0/m) * np.dot(dZ, layer.a_prev.T)
            db = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)
            dWs.append(dW)
            dBs.append(db)

            dA = np.dot(layer.weights.T, dZ)

        return dWs, dBs
    
    def update_parameters(self):
        pass

class Adam(Optimizer):
    def backward(self, loss_deriv, inp, labels, layers: list):
        dWs = []
        dBs = []
        dA = loss_deriv
        for layer in reversed(layers):
            dZ = dA * layer.activation.derivative(layer.z_cached)
            m = layer.a_prev.shape[1]
            # still missing exponentially weighted averages and rmsprop
            dW = (1.0/m) * np.dot(dZ, layer.a_prev.T)
            db = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)
            dWs.append(dW)
            dBs.append(db)

            dA = np.dot(layer.weights.T, dZ)

        return dWs, dBs
    
    def update_parameters(self, layers, dW, dB):
        for i, layer in enumerate(reversed(layers)):
            #momentum and rmsprop still missing 
            layer.weights -= learning_rate*dW[i]
            layer.biases -= learning_rate*dB[i]