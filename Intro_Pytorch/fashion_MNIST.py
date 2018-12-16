import torch
from torch import optim
from torchvision import datasets, transforms
from helper import *

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
# Load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

model = torch.nn.Sequential(torch.nn.Linear(784, 256),
                            torch.nn.ReLU(),
                            torch.nn.Linear(256, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 10),
                            torch.nn.LogSoftmax(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.003)

crit = torch.nn.NLLLoss()

epoch = 5
for _ in range(epoch):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], 784)
        legit = model(images)
        optimizer.zero_grad()
        loss = crit(legit, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")


dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)

# Plot the image and probabilities
view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
