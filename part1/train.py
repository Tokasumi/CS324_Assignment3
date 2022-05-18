import argparse
import itertools
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PalindromeDataset, OneHotPalindromeDataset
from lstm import RNN, LSTM

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["figure.dpi"] = 300

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)


def init_weights(module):
    module.apply(_init_weights)


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
    loss_records = []

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
            loss_records.append((step + 1, loss.item()))

        if step >= max_steps:
            break

    print('\nFit Complete!')
    return model, loss_records


def eval(model, data_loader, batch_size=1000):
    model.eval()

    input_data, labels = next(iter(data_loader))
    input_data, labels = input_data.to(DEVICE, torch.float32), labels.to(DEVICE, torch.float32)
    input_data = input_data.transpose(0, 1)

    if isinstance(model, RNN):
        output = forward_rnn(model, input_data, batch_size)
    elif isinstance(model, LSTM):
        output = forward_lstm(model, input_data, batch_size)
    else:
        raise TypeError

    predicts = torch.argmax(output, dim=1)
    labels = torch.argmax(labels, dim=1)
    confusion = predicts == labels
    acc = float(torch.sum(confusion).detach().to('cpu') / len(confusion))
    return acc


model_param = dict(input_dim=10, output_dim=10, hidden_dim=128)


def grid(model_name='lstm', seq_length=10, repeat=8, batch_size=128):
    lr_selection = [0.0001, 0.0005, 0.001, 0.0033, 0.0066, 0.01, 0.015, 0.02]
    data_loader = DataLoader(OneHotPalindromeDataset(seq_length + 1), batch_size=batch_size, num_workers=1)
    eval_loader = DataLoader(OneHotPalindromeDataset(seq_length + 1), batch_size=1000, num_workers=1)

    results = np.zeros((len(lr_selection), repeat), dtype=np.float64)
    for i, j in itertools.product(range(len(lr_selection)), range(repeat)):
        model = RNN(**model_param) if model_name == 'rnn' else LSTM(**model_param)
        model.to(DEVICE)
        fit(model, data_loader, batch_size=batch_size, max_steps=1000, use_adam=True, lr=lr_selection[i])
        acc = eval(model, eval_loader, batch_size=1000)
        print('STATS:', model_name, lr_selection[i], j, '%.2f' % acc)
        results[i, j] = acc

    best_lr = lr_selection[np.argmax(np.log2(results).mean(axis=0))]
    max_result = np.max(results)
    return best_lr, max_result


def ensemble():
    rnn_params = []
    rnn_best = []
    lstm_params = []
    lstm_best = []
    for seq_len in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        param, res = grid(model_name='rnn', seq_length=seq_len)
        rnn_params.append(param)
        rnn_best.append(res)
        param, res = grid(model_name='lstm', seq_length=seq_len)
        lstm_params.append(param)
        lstm_best.append(res)

    for lis in rnn_params, rnn_best, lstm_params, lstm_best:
        print(lis)


def train(config):
    seq_length = config.input_length
    batch_size = config.batch_size
    hidden_dim = config.num_hidden
    adam = config.adam
    learning_rate = config.learning_rate
    data_loader = DataLoader(OneHotPalindromeDataset(seq_length + 1), batch_size=batch_size, num_workers=1)
    test_loader = DataLoader(OneHotPalindromeDataset(seq_length + 1), batch_size=1000, num_workers=1)

    model = RNN(input_dim=10, output_dim=10, hidden_dim=hidden_dim).to(DEVICE)
    init_weights(model)
    print('=' * 30, 'RNN', '=' * 30)
    _, rnn_loss = fit(model, data_loader, batch_size=batch_size, max_steps=config.train_steps, lr=learning_rate,
                      use_adam=adam)
    print('Accuracy:', eval(model, data_loader=test_loader, batch_size=1000))
    model = LSTM(input_dim=10, output_dim=10, hidden_dim=hidden_dim).to(DEVICE)
    init_weights(model)
    print('=' * 30, 'LSTM', '=' * 30)
    _, lstm_loss = fit(model, data_loader, batch_size=batch_size, max_steps=config.train_steps, lr=learning_rate,
                       use_adam=adam)
    print('Accuracy:', eval(model, data_loader=test_loader, batch_size=1000))
    print('Done training.')

    plt.figure()
    plt.title(f'Sequence length: {seq_length}')
    plt.plot(*np.transpose(rnn_loss), linewidth=2.0, label='simple RNN')
    plt.plot(*np.transpose(lstm_loss), linewidth=2.0, label='simple LSTM')
    # plt.margins(0.1, 0.1)
    plt.grid()
    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'seqlen{seq_length}', dpi=300)


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
    parser.add_argument('--train-steps', type=int, default=1000, help='Number of training steps')
    # parser.add_argument('--max-norm', type=float, default=10.0)
    parser.add_argument('--adam', action='store_true', help='Use Adam as optimizer instead of RMSProp')

    return parser.parse_args()


if __name__ == "__main__":
    # ensemble()
    config = make_args()
    train(config)
