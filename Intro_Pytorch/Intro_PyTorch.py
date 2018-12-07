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
ans = torch.sum(features*weights) + bias
print('Answer without matrix multiplication is {}, time taken is {}.'.format(ans, time.time()-s))

a = time.time()
weights = weights.view(5, 1)
ans_mat = torch.sum(torch.mm(features, weights)) + bias
print('Answer without matrix multiplication is {}, time taken is {}.'.format(ans_mat, time.time()-a))
