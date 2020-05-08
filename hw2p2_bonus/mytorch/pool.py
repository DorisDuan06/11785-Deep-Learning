import numpy as np

class MaxPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

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
        B, C, W, H = x.shape
        W_p = (W - self.kernel) // self.stride + 1
        H_p = (H - self.kernel) // self.stride + 1
        out = np.zeros((B, C, W_p, H_p))
        for i in range(0, W - self.kernel + 1, self.stride):
            for j in range(0, H - self.kernel + 1, self.stride):
                out[:, :, i//self.stride, j//self.stride] = np.amax(x[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        B, C, W, H = self.x.shape
        self.dx = np.zeros(self.x.shape)
        count = np.zeros(self.x.shape)

        for i in range(0, W - self.kernel + 1, self.stride):
            for j in range(0, H - self.kernel + 1, self.stride):
                patch = self.x[:, :, i:i+self.kernel, j:j+self.kernel]
                self.dx[:, :, i:i+self.kernel, j:j+self.kernel] += (patch == np.amax(patch, axis=(2, 3), keepdims=True)) * delta[:, :, i//self.stride, j//self.stride].reshape(B, C, 1, 1)
                count[:, :, i:i+self.kernel, j:j+self.kernel] += (patch == np.amax(patch, axis=(2, 3), keepdims=True)) * np.ones((B, C, self.kernel, self.kernel))
        return np.where(count > 0, self.dx / count, self.dx)

class MeanPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

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
        B, C, W, H = x.shape
        W_p = (W - self.kernel) // self.stride + 1
        H_p = (H - self.kernel) // self.stride + 1
        out = np.zeros((B, C, W_p, H_p))
        for i in range(0, W - self.kernel + 1, self.stride):
            for j in range(0, H - self.kernel + 1, self.stride):
                out[:, :, i//self.stride, j//self.stride] = np.mean(x[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        B, C, W, H = self.x.shape
        self.dx = np.zeros(self.x.shape)
        mean_area = np.ones((B, C, self.kernel, self.kernel)) / self.kernel ** 2

        for i in range(0, W - self.kernel + 1, self.stride):
            for j in range(0, H - self.kernel + 1, self.stride):
                self.dx[:, :, i:i+self.kernel, j:j+self.kernel] += mean_area * delta[:, :, i//self.stride, j//self.stride].reshape(B, C, 1, 1)
        return self.dx
