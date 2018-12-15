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
    """
    This class contains the methods and attributes to build a Neural Network
    """
    def __init__(self):
        super().__init__()

        # First hidden layer
        self.hidden = nn.Linear(784, 128)
        # Second hidden layer
        self.hidden_1 = nn.Linear(128, 64)
        # Output layer
        self.output = nn.Linear(64, 10)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        This method is used to make 1 forward pass of the tensor x
        :param x: tensor to pass through all the layers
        :return: output tensor passed through all the layers
        """
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


# Way of checking weights and bias attached to a layer
model = Network()
print(model)
print(model.hidden_1.weight.shape)
print(model.hidden.weight)

#Set biases to all zeros
model.hidden.bias.data.fill_(0)
print(model.hidden.bias)

dataiter = iter(train_loader)
images, labels = next(dataiter)

images.resize_(64, 1, 28*28)

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx, :])

img = images[img_idx]
view_classify(img.view(1, 28, 28), ps)

# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(train_loader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
view_classify(images[0].view(1, 28, 28), ps)

# Creating a network using ordered dict

from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model

