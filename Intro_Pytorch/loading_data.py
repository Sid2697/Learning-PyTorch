import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from helper import *

data_dir = '/Users/siddhantbansal/PycharmProjects/Learning_PyTorch/Cat_Dog_data/'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                       ])

test_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                      ])
print('[INFO] Loading data..')
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(150528, 25000)
        self.fc2 = torch.nn.Linear(10000, 5000)
        self.fc3 = torch.nn.Linear(5000, 2000)
        self.fc4 = torch.nn.Linear(526, 256)
        self.fc5 = torch.nn.Linear(64, 10)
        self.fc6 = torch.nn.Linear(10, 2)

        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))

        x = F.log_softmax(self.dropout(self.fc6(x)), dim=1)

        return x


print('[INFO] Creating model...')
model = Network()

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
steps = 0

train_losses, test_losses = list(), list()

print('[INFO] Starting training...')
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

        with torch.no_grad():
            for images, labels in test_loader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloadTensor))

        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))

        print('Epoch: {}/{}...'.format(e+1, epochs),
              'Training Loss: {:.3f}...'.format(running_loss/len(train_loader)),
              'Test Loss: {:.3f}...'.format(test_loss/len(test_loader)),
              'Test Accuracy: {:.3f}...'.format(accuracy/len(test_loader)))

