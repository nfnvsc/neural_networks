import numpy as np

from layers.Dense import Dense
from Loss import Loss

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.learning_rate = 0
        self.loss = None

    def addLayer(self, layer):
        self.layers.append(layer)

    def computeLoss(self, y, y_hat):
        return self.loss.calculate(y, y_hat)
    
    def feedForward(self, features):
        out = features
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backPropagation(self, input, labels):
        dWs = []
        dBs = []

        dA = self.loss.derivative(labels, input)
        for layer in reversed(self.layers):
            dW, dB, dA = layer.backward(dA)
            dWs.append(dW)
            dBs.append(dB)

        return dWs, dBs        

    def updateParameters(self, dWs, dBs):
        for i, layer in enumerate(reversed(self.layers)):
            layer.update(dWs[i], dBs[i], self.learning_rate)
        
    def fit(self, X_train, Y_train, num_iterations=1000, learning_rate=0.0075, print_cost=False, loss_function="cross_entropy"):
        self.loss = Loss(loss_function)
        self.learning_rate = learning_rate
        
        cost = []

        for i in range(num_iterations):
            out = self.feedForward(X)
            loss = self.computeLoss(out, Y_train)

            cost.append(loss)

            if i % 100 == 0 and print_cost:
                print(f"Cost after {i} iterations: {loss}")

            dW, dB = self.backPropagation(out, Y)

            self.updateParameters(dW, dB)
        
        return cost



if __name__ == "__main__":
    import os
    import random
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    from sklearn.utils import shuffle

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

    print(Y)

    nn = NeuralNetwork()

    nn.addLayer(Dense(X.shape[0], 7, "relu"))
    nn.addLayer(Dense(7, 5, "relu"))
    nn.addLayer(Dense(5, 1, "sigmoid"))
    #nn.addLayer(Dense(7, 5, "relu"))
    #nn.addLayer(Dense(5, 1, "sigmoid"))

    cost = nn.fit(X, Y, num_iterations=3000, learning_rate=0.0075, print_cost=True)
    
    plt.plot(cost)
    plt.show()