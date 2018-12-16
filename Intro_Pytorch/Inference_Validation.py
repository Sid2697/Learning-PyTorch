from torchvision import datasets, transforms
from helper import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim, nn
import torch

# Define transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),)
                                ])
# Download and load the training data
train_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, transform=transform, train=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Download and load the test data
test_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


class Classifier(nn.Module):
    """This is the model that we'll train"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Flattening the input tensor
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        x = F.log_softmax(self.dropout(self.fc4(x)), dim=1)

        return x


# model = Classifier()
#
# images, labels = next(iter(test_loader))
# # Get the class probabilities
# ps = torch.exp(model(images))
# # print(ps.shape)
#
#
# top_p, top_class = ps.topk(1, dim=1)
# # print(top_class[:10, :])
#
# equals = top_class == labels.view(*top_class.shape)
#
# accuracy = torch.mean(equals.type(torch.FloatTensor))
# print(f'Accuracy: {accuracy.item()*100}%')

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
steps = 0

train_losses, test_losses = list(), list()

for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0

        # Turn of gradients
        with torch.no_grad():
            for images, labels in test_loader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))

        print('Epoch: {}/{}...'.format(e+1, epochs),
              'Training Loss: {:.3f}...'.format(running_loss/len(train_loader)),
              'Test Loss: {:.3f}...'.format(test_loss/len(test_loader)),
              'Test Accuracy: {:.3f}...'.format(accuracy/len(test_loader)))

# Plot the results
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

model.eval()

dataiter = iter(test_loader)
images, labels = dataiter.next()
img = images[0]

img = img.view(1, 784)

with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

view_classify(img.view(1, 28, 28), ps, version='Fashion')

print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())

torch.save(model.state_dict(), 'checkpoint.pth')
