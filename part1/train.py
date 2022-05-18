import argparse
import itertools
import logging
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
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
        cross_section = cross_section.unsqueeze(1) if cross_section.ndim == 1 else cross_section
        output, hidden_state = rnn(cross_section, hidden_state)
    return output


def forward_lstm(lstm: LSTM, batch_inputs, bsize):
    hidden_state = torch.autograd.Variable(torch.zeros(bsize, lstm.hidden_dim)).to(DEVICE, torch.float32)
    cell_state = torch.autograd.Variable(torch.zeros(bsize, lstm.hidden_dim)).to(DEVICE, torch.float32)
    output = None
    for cross_section in batch_inputs:
        cross_section = cross_section.unsqueeze(1) if cross_section.ndim == 1 else cross_section
        output, hidden_state, cell_state = lstm(cross_section, hidden_state, cell_state)
    return output


def fit(model, data_loader, max_steps=10000, eval_steps=10, batch_size=32, lr=0.01, use_adam=True):
    criterion = torch.nn.CrossEntropyLoss()

    if use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    model.train()
    back = model.state_dict().copy()
    loss_records = []
    last_loss = 100.00

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

        if step >= max_steps:
            break

        if (step + 1) % eval_steps == 0:
            if loss.item() > 2 * last_loss:
                model.load_state_dict(back)
                # logging.warning(f'\rROLLBACK, loss={loss.item()}')
                print('\r  ROLLBACK  ', end='')
                max_steps += 1
                continue
            print('\r%.2f%%' % ((step + 1) / max_steps * 100),
                  f'step:{step + 1}/{max_steps}',
                  'loss = %.6f' % loss.item(),
                  end='')

            loss_records.append((step + 1, loss.item()))
            last_loss = loss.item()
            back = model.state_dict().copy()

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


def train_task1():
    model = LSTM(input_dim=1, output_dim=10, hidden_dim=128).to(DEVICE)
    init_weights(model)
    data_loader = DataLoader(PalindromeDataset(10 + 1), batch_size=128, num_workers=1)
    test_loader = DataLoader(PalindromeDataset(10 + 1), batch_size=1000, num_workers=1)
    _, rmsp_loss = fit(model, data_loader, batch_size=128, max_steps=1000, lr=1e-3, use_adam=False)
    rmsp_acc = eval(model, test_loader, 1000)
    model = LSTM(input_dim=1, output_dim=10, hidden_dim=128).to(DEVICE)
    init_weights(model)
    _, adam_loss = fit(model, data_loader, batch_size=128, max_steps=1000, lr=1e-3, use_adam=True)
    adam_acc = eval(model, test_loader, 1000)

    plt.figure()
    plt.title(f'Task 1.1 LSTM training curve')

    plt.plot(*np.transpose(rmsp_loss), linewidth=1.5,
             label='LSTM with RMSProp, acc={:.2f}'.format(rmsp_acc))  # , color='#39c5bb')
    plt.plot(*np.transpose(adam_loss), linewidth=1.5,
             label='LSTM with Adam, acc={:.2f}'.format(adam_acc))  # , color='#e7acbb')
    # plt.margins(0.1, 0.1)
    plt.grid()
    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'lstm_test', dpi=300)


def transfer():
    max_steps = 1500

    lstm = LSTM(input_dim=1, output_dim=10, hidden_dim=128).to(DEVICE)
    init_weights(lstm)
    rnn = RNN(input_dim=1, output_dim=10, hidden_dim=128).to(DEVICE)
    init_weights(rnn)
    base_lr = 0.001

    for i in itertools.count(1):
        seq_len = i * 15
        base_lr = base_lr * 0.7
        print('=' * 50, seq_len, '=' * 50)

        data_loader = DataLoader(PalindromeDataset(seq_len + 1), batch_size=512, num_workers=1)
        test_loader = DataLoader(PalindromeDataset(seq_len + 1), batch_size=1024, num_workers=1)
        _, lstm_loss = fit(lstm, data_loader, batch_size=512, max_steps=max_steps, lr=base_lr * 0.9, use_adam=True)
        lstm_acc = eval(lstm, test_loader, 1024)
        _, rnn_loss = fit(lstm, data_loader, batch_size=512, max_steps=max_steps, lr=base_lr, use_adam=True)
        rnn_acc = eval(lstm, test_loader, 1024)

        plt.figure()
        plt.title(f'Sequence length: {seq_len}')
        plt.plot(*np.transpose(rnn_loss), linewidth=1.5,
                 label='simple RNN, acc={:.2f}'.format(rnn_acc))
        plt.plot(*np.transpose(lstm_loss), linewidth=1.5,
                 label='simple LSTM, acc={:.2f}'.format(lstm_acc))
        # plt.margins(0.1, 0.1)
        plt.grid()
        plt.xlabel('training steps')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'seqlen_transfer{seq_len}', dpi=300)


def train(config):
    seq_length = config.input_length
    batch_size = config.batch_size
    hidden_dim = config.num_hidden
    adam = config.adam
    learning_rate, learning_rate_lstm = config.learning_rate, config.learning_rate_lstm
    data_loader = DataLoader(PalindromeDataset(seq_length + 1), batch_size=batch_size, num_workers=1)
    test_loader = DataLoader(PalindromeDataset(seq_length + 1), batch_size=1000, num_workers=1)

    model = LSTM(input_dim=1, output_dim=10, hidden_dim=hidden_dim).to(DEVICE)
    init_weights(model)
    print('=' * 30, 'LSTM', '=' * 30)
    fit(model, data_loader=DataLoader(PalindromeDataset(seq_length // 2 + 1), batch_size=batch_size, num_workers=1),
        batch_size=batch_size, max_steps=config.train_steps + 1000, lr=learning_rate_lstm, use_adam=adam)
    _, lstm_loss = fit(model, data_loader, batch_size=batch_size, max_steps=config.train_steps, lr=learning_rate_lstm,
                       use_adam=adam)
    lstm_acc = eval(model, data_loader=test_loader, batch_size=1000)

    model = RNN(input_dim=1, output_dim=10, hidden_dim=hidden_dim).to(DEVICE)
    init_weights(model)
    print('=' * 30, 'RNN', '=' * 30)
    fit(model, data_loader=DataLoader(PalindromeDataset(seq_length // 2 + 1), batch_size=batch_size, num_workers=1),
        batch_size=batch_size, max_steps=config.train_steps, lr=learning_rate_lstm, use_adam=adam)
    _, rnn_loss = fit(model, data_loader, batch_size=batch_size, max_steps=config.train_steps, lr=learning_rate,
                      use_adam=adam)
    rnn_acc = eval(model, data_loader=test_loader, batch_size=1000)

    print('Done training.')

    plt.figure()
    plt.title(f'Sequence length: {seq_length}')
    plt.plot(*np.transpose(rnn_loss), linewidth=1.5, label='simple RNN, acc={:.2f}'.format(rnn_acc))
    plt.plot(*np.transpose(lstm_loss), linewidth=1.5, label='simple LSTM, acc={:.2f}'.format(lstm_acc))
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
    parser.add_argument('--learning-rate-lstm', type=float, default=0.001, help='Learning rate for LSTM')
    parser.add_argument('--train-steps', type=int, default=1000, help='Number of training steps')
    # parser.add_argument('--max-norm', type=float, default=10.0)
    parser.add_argument('--adam', action='store_true', help='Use Adam as optimizer instead of RMSProp')

    return parser.parse_args()


if __name__ == "__main__":
    # train_task1()
    try:
        transfer()
    except KeyboardInterrupt:
        print('Training Ended.')
    #
    # config = make_args()
    # train(config)
