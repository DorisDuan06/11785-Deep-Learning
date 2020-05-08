# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x
        batch_size, in_channel, input_size = x.shape
        output_size = (input_size - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channel, output_size))
        for i in range(0, input_size - self.kernel_size + 1, self.stride):
            out[:,:,i//self.stride] = np.tensordot(x[:,:,i:i+self.kernel_size], self.W, axes=([1,2],[1,2])) + self.b
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size, input_size = self.x.shape[0], self.x.shape[2]
        self.dx = np.zeros(self.x.shape)

        self.db = np.sum(delta, axis=(0, 2))
        for f in range(self.out_channel):
            for i in range(0, input_size - self.kernel_size + 1, self.stride):
                self.dW[f] += np.sum(self.x[:,:,i:i+self.kernel_size] * delta[:,f,i//self.stride].reshape(delta.shape[0], 1, 1), axis=0)

        for n in range(batch_size):
            for i in range(0, input_size - self.kernel_size + 1, self.stride):
                self.dx[n,:,i:i+self.kernel_size] += np.sum(self.W * delta[n,:,i//self.stride].reshape(self.out_channel, 1, 1), axis=0)

        return self.dx


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape(self.b, self.c, self.w)
