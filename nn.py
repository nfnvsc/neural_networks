import numpy as np
import pickle

from functions.Loss import Loss
from Optimizer import GradientDescent

class NeuralNetwork:
    def __init__(self, optimizer):
        self.layers = []
        self.learning_rate = 0
        self.loss = None    
        self.optimizer = optimizer

    def _feedForward(self, features):
        out = features
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _updateParameters(self, dWs, dBs):
        for i, layer in enumerate(reversed(self.layers)):
            layer.update(dWs[i], dBs[i], self.learning_rate)

    def addLayer(self, layer):
        self.layers.append(layer)    

    def fit(self, X_train, Y_train, num_iterations=1000, learning_rate=0.0075, print_cost=False, loss_function="cross_entropy"):
        self.loss = Loss(loss_function)
        self.learning_rate = learning_rate
        
        cost = []

        for i in range(num_iterations):
            pred = self._feedForward(X_train)
            loss = self.loss.calculate(pred, Y_train)

            cost.append(loss)

            if i % 100 == 0 and print_cost:
                print(f"Cost after {i} iterations: {loss}")
            
            loss_deriv = self.loss.derivative(pred, Y_train)

            dW, dB = self.optimizer.backward(loss_deriv, pred, Y_train, self.layers)

            self._updateParameters(dW, dB)
        
        return cost

    def predict(self, X):
        return self._feedForward(X)

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
    
    from layers.Dense import Dense

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
    X = np.array(X1).T
    Y = np.array(Y1).T

    print(X.shape)
    print(Y.shape)

    nn = NeuralNetwork(GradientDescent())

    nn.addLayer(Dense(X.shape[0], 7, "relu"))
    nn.addLayer(Dense(7, 5, "relu"))
    nn.addLayer(Dense(5, 1, "sigmoid"))
    #nn.addLayer(Dense(7, 5, "relu"))
    #nn.addLayer(Dense(5, 1, "sigmoid"))
    
    nn.load("test.pickle")
    cost = nn.fit(X, Y, num_iterations=1000, learning_rate=0.0075, print_cost=True)
    nn.save("test.pickle")
    plt.plot(cost)
    plt.show()