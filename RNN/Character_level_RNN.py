import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# Loading the data
with open('/Users/siddhantbansal/Desktop/Work/deep-learning-v2-pytorch/recurrent-neural-networks/char-rnn/data/anna.txt') as f:
    text = f.read()

# print(text[:100])

# Encoding the text and map each character to an integer and vice versa
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode the text
encoded = np.array([char2int[ch] for ch in text])

# print(encoded[:100])


def one_hot_encode(arr, n_labels):
    # initialize the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


test_seq = np.array([[3, 5, 1]])
one_hot = one_hot_encode(test_seq, 8)

# print(one_hot)


def get_batches(arr, batch_size, seq_length):
    n_batches = len(arr) // (batch_size * seq_length)
    arr = arr[:n_batches * batch_size * seq_length]
    arr = arr.reshape(batch_size, -1)
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


# Testing the implementation
batches = get_batches(encoded, 8, 50)
x, y = next(batches)

# printing our the first 10 items in the sequence
# print('x\n', x[:10, :10])
# print('\ny\n', y[:10, :10])

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU!')
else:
    print('Training on CPU!')


class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # Defining the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob,
                            batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        # get the output and the hidden state from the LSTM
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)

        # stack up LSTM outputs
        out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden state
        """
        weight = next(self.parameters()).data
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=5):
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Creating training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if train_on_gpu:
        net.cuda()

    counter = 0
    n_chars = len(net.chars)

    for e in range(epochs):
        # initialize the hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # one hot encode the data and make them torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # creating new variables for the hidden state otherwise we'll
            # backprop through the entire training history
            h = tuple([each.data for each in h])

            net.zero_grad()
            output, h = net(inputs, h)

            # calculating the loss and performing backprop
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            if counter % print_every == 0:
                # get the validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # one hot encode the data
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length))

                    val_losses.append(val_loss.item())

                net.train()

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


n_hidden = 512
n_layers = 2

net = CharRNN(chars, n_hidden, n_layers)
print(net)

batch_size = 128
seq_length = 100
n_epochs = 2

train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001)

model_name = 'rnn_20_epoch.net'
checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)


def predict(net, char, h=None, top_k=None):
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if train_on_gpu:
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    out, h = net(inputs, h)

    p = F.softmax(out, dim=1).data
    if train_on_gpu:
        p = p.cpu()

    # get top chars
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):

    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()

    net.eval()

    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


print(sample(net, 1000, prime='Anna', top_k=5))

# loading a checkpoint
with open('rnn_20_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

# sample using a loaded model
print(sample(loaded, 2000, top_k=5, prime='And Levin said'))

