import numpy as np
import random

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000,addBias=True,mse_threshold=1):
        self.weights = np.random.rand(2) #2 deh number of features   [w1 w2]
        self.lr = learning_rate
        self.ep = epochs
        self.mse_threshold = mse_threshold
        self.activation_fn = self._signum
        self.addBias = addBias
        self.line_outputs = []
        if(self.addBias==True): self.bias = 0.0001
        else: self.bias= random.uniform(0.0, 1.0)

    def train(self, X, Y):
        mse_list = []
        for _ in range(self.ep):
            for i, x_ in enumerate(X):                
                output = np.dot(x_, self.weights) + self.bias
                y_pred = self.activation_fn(output)
                error = self.lr * (Y[i] - y_pred)
                self.weights =self.weights + error * x_
                if(self.addBias==True): self.bias = self.bias + error 
                else: self.bias=0
            mse = np.mean((Y - np.dot(X, self.weights)) ** 2)
            if mse < self.mse_threshold:
                break
            mse_list.append(mse)

    def test(self, X):
        y_pred = []
        for x_ in X:
            output = np.dot(x_, self.weights) + self.bias
            self.line_outputs.append(output)
            y_pred.append(self.activation_fn(output))

        return y_pred

    def _signum(self, x):
        if x > 0:
            return 1
        elif x <= 0:
            return -1


