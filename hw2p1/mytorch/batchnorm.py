# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        if eval:
            return self.gamma * (x - self.running_mean) / np.sqrt(self.running_var + self.eps) + self.beta

        self.x = x

        self.mean = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.invstd = 1 / np.sqrt(self.var + self.eps)
        self.norm = (x - self.mean) * self.invstd
        self.out = self.gamma * self.norm + self.beta

        # Update running batch statistics
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """

        # cache = (x_hat, sample_mean, xmean, gamma, invstd, eps)

        N = delta.shape[0]
        self.dgamma = np.sum(delta * self.norm, axis=0, keepdims=True)
        self.dbeta = np.sum(delta, axis=0, keepdims=True)
        out = (N * delta - self.dbeta - self.norm * self.dgamma) * self.gamma * self.invstd / N
        return out
