import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset, OneHotPalindromeDataset
from lstm import RNN, LSTM

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def forward_rnn(rnn: RNN, batch_inputs, bsize):
    hidden_state = torch.autograd.Variable(torch.zeros(bsize, rnn.hidden_dim)).to(DEVICE, torch.float32)
    output = None
    for cross_section in batch_inputs:
        output, hidden_state = rnn(cross_section, hidden_state)
    return output


def forward_lstm(lstm: LSTM, batch_inputs, bsize):
    hidden_state = torch.autograd.Variable(torch.zeros(bsize, lstm.hidden_dim)).to(DEVICE, torch.float32)
    cell_state = torch.autograd.Variable(torch.zeros(bsize, lstm.hidden_dim)).to(DEVICE, torch.float32)
    output = None
    for cross_section in batch_inputs:
        output, hidden_state, cell_state = lstm(cross_section, hidden_state, cell_state)
    return output


def fit(model, data_loader, max_steps=10000, eval_steps=10, batch_size=32, lr=0.01, use_adam=True):
    criterion = torch.nn.CrossEntropyLoss()

    if use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    model.train()

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(DEVICE, torch.float32)
        batch_targets = batch_targets.to(DEVICE, torch.float32)
        batch_inputs = batch_inputs.transpose(0, 1)  # [batch, len, feats] -> [len, batch, feats]
        if isinstance(model, RNN):
            output = forward_rnn(model, batch_inputs, batch_size)
        elif isinstance(model, LSTM):
            output = forward_lstm(model, batch_inputs, batch_size)
        else:
            raise TypeError

        loss = criterion(output, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % eval_steps == 0:
            print('\r%.2f%%' % ((step + 1) / max_steps * 100),
                  f'step:{step + 1}/{max_steps}',
                  'loss = %.6f' % loss.item(),
                  end='')

        if step >= max_steps:
            break

    print('\nFit Complete!')
    return model


def train(config):
    # Initialize the model that we are going to use
    model = None  # fixme

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = None  # fixme
    optimizer = None  # fixme

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Add more code here ...

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

        # Add more code here ...

        loss = np.inf  # fixme
        accuracy = 0.0  # fixme

        if step % 10 == 0:
            pass
            # print acuracy/loss here

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


def make_args():
    # Parse training configuration
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of o_gate sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of prev_hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    dataset = OneHotPalindromeDataset(10 + 1)
    batch_size = 64
    hidden_size = 30
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    model = RNN(input_dim=10, output_dim=10, hidden_dim=hidden_size).to(DEVICE)
    print('=' * 30, 'RNN', '=' * 30)
    fit(model, data_loader, batch_size=batch_size, use_adam=False)
    model = LSTM(input_dim=10, output_dim=10, hidden_dim=hidden_size).to(DEVICE)
    print('=' * 30, 'LSTM', '=' * 30)
    fit(model, data_loader, batch_size=batch_size, use_adam=False)
    # config = make_args()
    # train(config)
