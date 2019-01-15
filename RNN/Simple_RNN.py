import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

# How many time steps/data pts are in one batch of data
seq_len = 20

time_steps = np.linspace(0, np.pi, seq_len + 1)
data = np.sin(time_steps)
data.resize((seq_len + 1, 1))  # Add an input dimension

x = data[:-1]  # All but the last piece of data
y = data[1:]  # All but the first piece of data

# display the data
# plt.plot(time_steps[1:], x, 'r.', label='input, x')
# plt.plot(time_steps[1:], y, 'b.', label='target, y')
#
# plt.legend(loc='best')
# plt.show()


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # Define an RNN with specified parameters
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # Last fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # Get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape the output to be (batch_size * seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)

        # get the final output
        output = self.fc(r_out)

        return output, hidden


# checking the input and output dimensions
test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)

# generate evenly spaced, test data pts
time_steps = np.linspace(0, np.pi, seq_len)
data = np.sin(time_steps)
data.resize((seq_len, 1))

test_input = torch.Tensor(data).unsqueeze(0)
# print('Input size', test_input.size())

# test the RNN sizes
test_out, test_h = test_rnn(test_input, None)
# print('Output size', test_out.size())
# print('Hidden state size: ', test_h.size())

# Training the RNN

# deciding the hyperparameters
input_size = 1
output_size = 1
hidden_dim = 32
n_layers = 1

# Instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


def train(rnn, n_steps, print_every):
    # Initialize the training data
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data
        time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_len + 1)
        data = np.sin(time_steps)
        data.resize((seq_len + 1, 1))

        x = data[:-1]
        y = data[1:]

        # convert the data into tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)
        y_tensor = torch.Tensor(y)

        # output data from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        # making a new variable for hidden and detach the hidden state from its history
        # this way, we do not backpropagate through entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and prediction
        if batch_i % print_every == 0:
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.')
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')
            plt.show()

    return rnn


# train the rnn and monitor results
n_steps = 1000
print_every = 100

trained_rnn = train(rnn, n_steps, print_every)
