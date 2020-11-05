from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Interface for optimizers
    """
    @abstractmethod
    def backward(self, loss_deriv, inp, labels, layers: list, num_iter):
        pass

    @abstractmethod
    def update_parameters(self, layers: list, dW, dB):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def backward(self, loss_deriv, inp, labels, layers: list, num_iter=0):
        dW = []
        dB = []
        da = loss_deriv
        for layer in reversed(layers):
            dz = da * layer.activation.derivative(layer.z_cached)
            m = layer.a_prev.shape[1]

            dw = (1.0/m) * np.dot(dz, layer.a_prev.T)
            db = (1.0/m) * np.sum(dz, axis=1, keepdims=True)
            print(layer.w)
            
            print(dw)

            dW.append(dw)
            dB.append(db)

            da = np.dot(layer.w.T, dz)

        self.update_parameters(layers, dW, dB)
    
    def update_parameters(self, layers: list, dW, dB):
        for i, layer in enumerate(reversed(layers)):
            layer.w -= self.lr*dW[i]
            layer.b -= self.lr*dB[i]


class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-1):
        self.lr = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.e = epsilon

    def backward(self, loss_deriv, inp, labels, layers: list, num_iter):
        dW = []
        dB = []

        da = loss_deriv
        for layer in reversed(layers):
            dz = da * layer.activation.derivative(layer.z_cached)
            m = layer.a_prev.shape[1]
            
            dw = (1.0/m) * np.dot(dz, layer.a_prev.T)
            db = (1.0/m) * np.sum(dz, axis=1, keepdims=True)

            da = np.dot(layer.w.T, dz)

            #exponetial weigthed averages
            layer.vdw = np.multiply(self.b1, layer.vdw) + np.multiply(1 - self.b1, dw)
            layer.vdb = np.multiply(self.b1, layer.vdb) + np.multiply(1 - self.b1, db)
            
            #rmsprop 
            layer.sdw = np.multiply(self.b2, layer.sdw) + np.multiply(1 - self.b2, np.power(dw, 2))
            layer.sdb = np.multiply(self.b2, layer.sdb) + np.multiply(1 - self.b2, np.power(db, 2))

            #bias normalization
            vdw_corrected = np.divide(layer.vdw, 1 - np.power(self.b1, num_iter))
            vdb_corrected = np.divide(layer.vdb, 1 - np.power(self.b1, num_iter))

            sdw_corrected = np.divide(layer.sdw, 1 - np.power(self.b2, num_iter))
            sdb_corrected = np.divide(layer.sdb, 1 - np.power(self.b2, num_iter))

            dw = np.divide(vdw_corrected, np.sqrt(sdw_corrected) + self.e)
            db = np.divide(vdb_corrected, np.sqrt(sdb_corrected) + self.e)

            dW.append(dw)
            dB.append(db)


        self.update_parameters(layers, dW, dB)

    def update_parameters(self, layers, dW, dB):
        for i, layer in enumerate(reversed(layers)):
            layer.w -= self.lr*dW[i]
            layer.b -= self.lr*dB[i]