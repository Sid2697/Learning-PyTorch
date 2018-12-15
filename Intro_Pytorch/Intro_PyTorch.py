"""
This file contains the code written while learning about PyTorch.
"""
# Importing PyTorch
import torch
import time


def sigmoid_activation(x):
    """
    Sigmoid Activation function
    :param x: torch.Tensor
    :return: sigmodial value of x
    """
    return 1/(1 + torch.exp(-x))


# Generating some data
torch.manual_seed(7)  # Setting the random seed
# Features
features = torch.randn((1, 5))
# Random weights
weights = torch.rand_like(features)
# Bias value
bias = torch.randn((1, 1))

s = time.time()
ans = sigmoid_activation(torch.sum(features*weights) + bias)
print('Answer without matrix multiplication is {}, time taken is {}.'.format(ans, time.time()-s))

a = time.time()
weights = weights.view(5, 1)
ans_mat = sigmoid_activation(torch.mm(features, weights) + bias)
print('Answer without matrix multiplication is {}, time taken is {}.'.format(ans_mat, time.time()-a))

# Generate some data
torch.manual_seed(7)  # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

# Your solution here
a = sigmoid_activation(torch.mm(features, W1) + B1)
b = sigmoid_activation(torch.mm(a, W2) + B2)
print('Value of a small neural network is {}.'.format(b))
