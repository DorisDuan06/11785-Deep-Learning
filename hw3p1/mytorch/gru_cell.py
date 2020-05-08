import numpy as np
from activation import *

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wzh = np.random.randn(h, h)
        self.Wrh = np.random.randn(h, h)
        self.Wh  = np.random.randn(h, h)

        self.Wzx = np.random.randn(h, d)
        self.Wrx = np.random.randn(h, d)
        self.Wx  = np.random.randn(h, d)

        self.dWzh = np.zeros((h, h))
        self.dWrh = np.zeros((h, h))
        self.dWh  = np.zeros((h, h))

        self.dWzx = np.zeros((h, d))
        self.dWrx = np.zeros((h, d))
        self.dWx  = np.zeros((h, d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.z = self.z_act(np.dot(self.Wzh, self.hidden) + np.dot(self.Wzx, self.x))
        self.r = self.r_act(np.dot(self.Wrh, self.hidden) + np.dot(self.Wrx, self.x))
        self.h_tilda = self.h_act(np.dot(self.Wh, self.r * self.hidden) + np.dot(self.Wx, self.x))
        h_t = (1 - self.z) * self.hidden + self.z * self.h_tilda

        assert self.x.shape == (self.d, )
        assert self.hidden.shape == (self.h, )

        assert self.r.shape == (self.h, )
        assert self.z.shape == (self.h, )
        assert self.h_tilda.shape == (self.h, )
        assert h_t.shape == (self.h, )

        return h_t


    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        x = self.x.reshape(1, self.d)
        hidden = self.hidden.reshape(1, self.h)
        r = self.r.reshape(1, self.h)
        z = self.z.reshape(1, self.h)
        h_tilda = self.h_tilda.reshape(1, self.h)

        dz_t = -delta * hidden + delta * h_tilda  # 1 x h
        dh_tilda = delta * z  # 1 x h
        dh = delta * (1 - z)  # 1 x h

        dh_act = dh_tilda * self.h_act.derivative()  # 1 x h
        self.dWx = np.dot(dh_act.T, x)  # h x d
        self.dWh = np.dot(dh_act.T, r * hidden)  # h x h
        dr_t = np.dot(dh_act, self.Wh) * hidden  # 1 x h
        dh += np.dot(dh_act, self.Wh) * r  # 1 x h
        dx = np.dot(dh_act, self.Wx)  # 1 x d

        dr_act = dr_t * self.r_act.derivative()  # 1 x h
        self.dWrh = np.dot(dr_act.T, hidden)  # h x h
        self.dWrx = np.dot(dr_act.T, x)  # h x d
        dh += np.dot(dr_act, self.Wrh)  # 1 x h
        dx += np.dot(dr_act, self.Wrx)  # 1 x d

        dz_act = dz_t * self.z_act.derivative()  # 1 x h
        self.dWzh = np.dot(dz_act.T, hidden)  # h x h
        self.dWzx = np.dot(dz_act.T, x)  # h x d
        dh += np.dot(dz_act, self.Wzh)  # 1 x h
        dx += np.dot(dz_act, self.Wzx)  # 1 x d

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
