import numpy as np

from layers.Dense import Dense
from Loss import Loss

C = 1e-7

class NeuralNetwork:
    def __init__(self, learning_rate, loss_function):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss = Loss(loss_function)

    def addLayer(self, layer):
        self.layers.append(layer)

    def computeLoss(self, y, y_hat):
        return self.loss.calculate(y, y_hat)
    
    def feedForward(self, features, labels):
        out = features
        for layer in self.layers:
            out = layer.forward(out)

        loss = self.computeLoss(out, labels)
        return out, loss

    def backPropagation(self, input, labels):
        dWs = []
        dBs = []

        dA = self.loss.derivative(labels, input)
        for layer in reversed(self.layers):
            dW, dB, dA = layer.backward(dA, input.shape[1])
            dWs.append(dW)
            dBs.append(dB)

        return dWs, dBs        

    def updateParameters(self, dWs, dBs):
        for i, layer in enumerate(reversed(self.layers)):
            layer.update(dWs[i], dBs[i], self.learning_rate)



if __name__ == "__main__":
    import os
    import random
    from PIL import Image, ImageOps
    import matplotlib.pyplot as plt

    num_iterations = 10
    learning_rate = 0.075

    X = []
    Y = []

    TRAIN_DIR = "train"

    for d in os.listdir(TRAIN_DIR):
        mx = 0
        for f in os.listdir(os.path.join(TRAIN_DIR, d)):
            if mx == 100:
                break
            mx += 1
            try:
                img = Image.open(os.path.join(TRAIN_DIR, d, f))
                img = img.resize((64,64))
                img = ImageOps.grayscale(img)
                X.append(np.asarray(img).reshape(-1))
                Y.append(1 if d == "Cat" else 0)
            except Exception:
                pass
    
    X = np.array(X).T
    Y = np.array(Y).reshape(1,-1)
    
    print(X.shape)
    print(Y.shape)

    nn = NeuralNetwork(learning_rate, "cross_entropy")

    nn.addLayer(Dense(X.shape[0], 20, "sigmoid"))
    nn.addLayer(Dense(20, 7, "sigmoid"))
    nn.addLayer(Dense(7, 5, "sigmoid"))
    nn.addLayer(Dense(5, 1, "sigmoid"))

    LOSS = []

    for _ in range(num_iterations):
        out, loss = nn.feedForward(X, Y)
        LOSS.append(loss)
        dW, dB = nn.backPropagation(out, Y)
        nn.updateParameters(dW, dB)

    plt.plot(LOSS)
    plt.show()