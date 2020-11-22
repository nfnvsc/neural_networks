import numpy as np
import pickle

import core.config as config

from core.layers.Layer import Layer
from core.functions.Loss import Loss
from core.functions.Activation import Activation
from core.optimizer.Optimizer import Optimizer, Adam

class Dataset:
    pass

class NeuralNetwork:
    def __init__(self, layers=None):
        self.layers = []

    def add(self, layer):
        """
        Add a layer to the network
        """
        if isinstance(layer, Layer):
            self.layers.append(layer)    
        else:
            raise TypeError(f"{type(layer)} is not a {Layer} instance.")
        
    def compile(self, optimizer, loss):
        """
        Add Loss and Optimizer to the network
        """
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise TypeError(f"{type(optimizer)} is not a {type(Optimizer)} instance.")

        if isinstance(loss, Loss):
            self.loss = loss
        else:
            raise TypeError(f"{type(loss)} is not a {type(Loss)} instance.")


    def fit(self, X_train, Y_train, num_iterations=1000, print_cost=False):
        """
        Fit the model with values
        """

        cost = []

        for i in range(num_iterations):
            prediction = self.predict(X_train)
            loss = self.loss.calculate(prediction, Y_train)

            cost.append(loss)

            if i % config.COST_EVERY_N_ITERATIONS == 0 and print_cost:
                print(f"Cost after {i} iterations: {loss}")
            
            loss_deriv = self.loss.derivative(prediction, Y_train)

            self.optimizer.backward(loss_deriv, prediction, self.layers)
        
        return cost
    
    def predict(self, features):
        out = features
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def load(self, file):
        saved = pickle.load(open(file, "rb"))

        for i, layer in enumerate(self.layers):
            layer.load(saved[i][0], saved[i][1])

    def save(self, file):
        saved = []

        for layer in self.layers:
            saved.append(layer.save())

        pickle.dump(saved, open(file, "wb"))


if __name__ == "__main__":
    import os
    import random
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    from sklearn.utils import shuffle
    
    from core.layers.Dense import Dense

    X = []
    Y = []

    TRAIN_DIR = "train"

    for d in os.listdir(TRAIN_DIR):
        mx = 0
        for f in os.listdir(os.path.join(TRAIN_DIR, d)):
            if mx == 200:
                break
            mx += 1
            try:
                img = Image.open(os.path.join(TRAIN_DIR, d, f))
                img = img.resize((64,64))
                #img = ImageOps.grayscale(img)
                X.append(np.asarray(img).reshape(12288))
                Y.append(1 if d == "Cat" else 0)
            except Exception:
                pass
    
    X = np.array(X)/255.0
    Y = np.array(Y).reshape(1,-1)

    X1, Y1 = shuffle(X, Y.T)
    X_train = np.array(X1).T
    Y_train = np.array(Y1).T

    print(X.shape)
    print(Y.shape)

    #hyperparameters
    learning_rate = 0.01

    opt = Adam(learning_rate)
    loss = Loss("cross_entropy")

    nn = NeuralNetwork()
    nn.add(Dense(X_train.shape[0], 20, "relu"))
    nn.add(Dense(20, 7, "relu"))
    nn.add(Dense(7, 5, "relu"))
    nn.add(Dense(5, 1, "sigmoid"))
    nn.compile(opt, loss)
    cost = nn.fit(X_train, Y_train, num_iterations=10000, print_cost=True)
    nn.save("test.pickle")

    plt.plot(cost)
    plt.show()