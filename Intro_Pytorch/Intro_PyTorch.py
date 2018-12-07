"""
This file contains the code written while learning about PyTorch.
"""
# Importing PyTorch
import torch


def sigmoid_activation(x):
    """
    Sigmoid Activation function
    :param x: torch.Tensor
    :return: sigmodial value of x
    """
    return 1/(1 + torch.exp(-x))

