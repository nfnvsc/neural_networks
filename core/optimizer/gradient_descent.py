class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def backward(self, loss_deriv, inp, layers: list):
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