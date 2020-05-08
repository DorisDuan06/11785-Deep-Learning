"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.linear_layers = []
        if self.nlayers == 1:
            self.linear_layers.append(Linear(self.input_size, self.output_size, weight_init_fn, bias_init_fn))
        else:
            for i in range(self.nlayers):
                if i == 0:
                    self.linear_layers.append(Linear(self.input_size, hiddens[i], weight_init_fn, bias_init_fn))
                elif i == self.nlayers - 1:
                    self.linear_layers.append(Linear(hiddens[i-1], self.output_size, weight_init_fn, bias_init_fn))
                else:
                    self.linear_layers.append(Linear(hiddens[i-1], hiddens[i], weight_init_fn, bias_init_fn))

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = []
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(hiddens[i]))

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        self.output = x
        for i in range(len(self.linear_layers)):
            self.output = self.linear_layers[i](self.output)
            if i < self.num_bn_layers:
                self.output = self.bn_layers[i](self.output, not self.train_mode)
            self.output = self.activations[i](self.output)
        return self.output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)
            if i < self.num_bn_layers:
                self.bn_layers[i].dgamma.fill(0.0)
                self.bn_layers[i].dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
            self.linear_layers[i].W += self.linear_layers[i].momentum_W
            self.linear_layers[i].momentum_b = self.momentum * self.linear_layers[i].momentum_b - self.lr * self.linear_layers[i].db
            self.linear_layers[i].b += self.linear_layers[i].momentum_b

        # Do the same for batchnorm layers
        for i in range(self.num_bn_layers):
            self.bn_layers[i].gamma -= self.lr * self.bn_layers[i].dgamma
            self.bn_layers[i].beta -= self.lr * self.bn_layers[i].dbeta

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        self.loss = self.criterion(self.output, labels)
        dout = self.criterion.derivative()
        for i in range(len(self.linear_layers)-1, -1, -1):
            dout = dout * self.activations[i].derivative()
            if i < self.num_bn_layers:
                dout = self.bn_layers[i].backward(dout)
            dout = self.linear_layers[i].backward(dout)

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...
        np.random.shuffle(idxs)

        mlp.train()
        train_loss, train_error = 0, 0
        for b in range(0, len(trainx), batch_size):
            # Train ...
            mlp.zero_grads()
            mlp.forward(trainx[idxs[b:b + batch_size]])
            mlp.backward(trainy[idxs[b:b + batch_size]])
            mlp.step()
            train_loss += mlp.total_loss(trainy[idxs[b:b + batch_size]])
            train_error += mlp.error(trainy[idxs[b:b + batch_size]])

        mlp.eval()
        val_loss, val_error = 0, 0
        for b in range(0, len(valx), batch_size):
            # Val ...
            mlp.forward(valx[b:b + batch_size])
            val_loss += mlp.total_loss(valy[b:b + batch_size])
            val_error += mlp.error(valy[b:b + batch_size])

        # Accumulate data...
        training_losses[e] += train_loss / (len(trainx) // batch_size * batch_size)
        training_errors[e] += train_error / (len(trainx) // batch_size * batch_size)
        validation_losses[e] += val_loss / (len(valx) // batch_size * batch_size)
        validation_errors[e] += val_error / (len(valx) // batch_size * batch_size)

    # Return results ...
    return (training_losses, training_errors, validation_losses, validation_errors)
