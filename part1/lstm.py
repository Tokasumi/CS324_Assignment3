import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.concat_dim = input_dim + hidden_dim
        self.output_dim = output_dim

        self.input_to_hidden = nn.Sequential(
            nn.Linear(self.concat_dim, self.hidden_dim, bias=True),
            nn.Tanh()
        )
        self.hidden_to_out = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, current_input, prev_hidden):
        concatenate = torch.cat((current_input, prev_hidden), 1)
        current_hidden = self.input_to_hidden(concatenate)
        current_output = self.hidden_to_out(current_hidden)
        current_output = current_output if self.training else F.softmax(current_output, -1)
        return current_output, current_hidden


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = hidden_dim
        self.output_dim = output_dim

        self.f_gate = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.cell_dim, bias=True),
            nn.Sigmoid()
        )
        self.i_gate = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.cell_dim, bias=True),
            nn.Sigmoid()
        )
        self.g_gate = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.cell_dim, bias=True),
            nn.Tanh()
        )
        self.o_gate = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.cell_dim, bias=True),
            nn.Sigmoid()
        )

        self.hidden_to_out = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, current_input, prev_hidden, prev_cell):
        concatenate = torch.cat((current_input, prev_hidden), 1)

        f = self.f_gate(concatenate)
        i = self.i_gate(concatenate)
        g = self.g_gate(concatenate)
        o = self.o_gate(concatenate)

        current_cell = g * i + prev_cell * f
        current_hidden = o * self.tanh(current_cell)
        current_output = self.hidden_to_out(current_hidden)
        current_output = current_output if self.training else F.softmax(current_output, -1)

        return current_output, current_hidden, current_cell
