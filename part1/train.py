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
    seq_length = config.input_length
    batch_size = config.batch_size
    hidden_dim = config.num_hidden
    adam = config.adam
    learning_rate = config.learning_rate
    data_loader = DataLoader(OneHotPalindromeDataset(seq_length + 1), batch_size=batch_size, num_workers=1)
    model = RNN(input_dim=10, output_dim=10, hidden_dim=hidden_dim).to(DEVICE)
    print('=' * 30, 'RNN', '=' * 30)
    fit(model, data_loader, batch_size=batch_size, max_steps=config.train_steps, lr=learning_rate, use_adam=adam)
    model = LSTM(input_dim=10, output_dim=10, hidden_dim=hidden_dim).to(DEVICE)
    print('=' * 30, 'LSTM', '=' * 30)
    fit(model, data_loader, batch_size=batch_size, max_steps=config.train_steps, lr=learning_rate, use_adam=adam)
    print('Done training.')


def make_args():
    # Parse training configuration
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--input-length', type=int, default=10, help='Length of an input sequence')
    # parser.add_argument('--input-dim', type=int, default=10, help='Dimensionality of input sequence')
    # parser.add_argument('--num-classes', type=int, default=10, help='Dimensionality of o_gate sequence')
    parser.add_argument('--num-hidden', type=int, default=128, help='Number of prev_hidden units in the model')
    parser.add_argument('--batch-size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train-steps', type=int, default=10000, help='Number of training steps')
    # parser.add_argument('--max-norm', type=float, default=10.0)
    parser.add_argument('--adam', action='store_true', help='Use Adam as optimizer instead of RMSProp')

    return parser.parse_args()


if __name__ == "__main__":
    config = make_args()
    train(config)
