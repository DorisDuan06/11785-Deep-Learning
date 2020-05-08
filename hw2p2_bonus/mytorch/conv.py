import numpy as np

class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

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
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        batch_size, in_channel, W, H = x.shape
        W_p = (W - self.kernel_size) // self.stride + 1
        H_p = (H - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channel, W_p, H_p))
        for b in range(batch_size):
            for i in range(0, W - self.kernel_size + 1, self.stride):
                for j in range(0, H - self.kernel_size + 1, self.stride):
                    out[b, :, i//self.stride, j//self.stride] = np.sum(self.W * x[b, :, i:i+self.kernel_size, j:j+self.kernel_size], axis=(1,2,3)) + self.b
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, out_channel, W, H = self.x.shape
        self.dx = np.zeros(self.x.shape)

        self.db = np.sum(delta, axis=(0, 2, 3))
        for f in range(self.out_channel):
            for i in range(0, W - self.kernel_size + 1, self.stride):
                for j in range(0, H - self.kernel_size + 1, self.stride):
                    self.dW[f] += np.sum(self.x[:,:,i:i+self.kernel_size, j:j+self.kernel_size] * delta[:, f, i//self.stride, j//self.stride].reshape(delta.shape[0], 1, 1, 1), axis=0)

        for n in range(batch_size):
            for i in range(0, W - self.kernel_size + 1, self.stride):
                for j in range(0, H - self.kernel_size + 1, self.stride):
                    self.dx[n, :, i:i+self.kernel_size, j:j+self.kernel_size] += np.sum(self.W * delta[n, :, i//self.stride, j//self.stride].reshape(self.out_channel, 1, 1, 1), axis=0)
        return self.dx
