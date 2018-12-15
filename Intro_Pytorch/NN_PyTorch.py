# Importing necessary packages
import numpy as np
import torch
from helper import *
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
"""
This file contains the code to make simple Neural Networks using PyTorch
"""
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
print('[INFO] Loading MNIST training data')
train_set = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

data_iter = iter(train_loader)
images, labels = data_iter.__next__()
print(type(images))
print(images.shape)
print(labels.shape)

images = images.view(64, 784)

n_input = images.shape[1]
n_hidden = 256
n_output = 10

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

a = sigmoid_activation(torch.mm(images, W1) + B1)
b = torch.mm(a, W2) + B2
print(a.shape)
print(b.shape)


def softmax(x):
    """
    This function gives the output value between 0 and 1
    :param x: input value
    :return: value between 0 and 1
    """
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)


probabilities = softmax(b)
print(probabilities.shape)
print(probabilities.sum(dim=1))

# Building networks with PyTorch
from torch import nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        This method passes the input tensor through each of our operation
        :param x: tensor to pass through all the layers
        :return: output tensor from each layer
        """
        x = self.hidden(x)
        x = self.softmax(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


