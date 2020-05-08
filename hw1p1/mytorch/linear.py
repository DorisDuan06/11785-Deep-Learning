# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_b (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.momentum_W = np.zeros_like(self.W)
        self.momentum_b = np.zeros_like(self.b)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """

        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """

        N, D = delta.shape
        self.dW = np.dot(self.x.T, delta) / N
        self.db = np.sum(delta, axis=0).reshape(1, D) / N
        self.dx = np.dot(delta, self.W.T)
        return self.dx
