import torch.nn as nn
from torch.nn.utils.rnn import *


class RNN(nn.Module):
    def __init__(self, in_vocab, out_vocab, hidden_size, n_layers=1):
        super(RNN1, self).__init__()
        self.cnns = nn.Sequential(
                        nn.Conv1d(40, 128, 3, 1, 1, bias=False),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(128),
                        nn.Conv1d(128, 256, 3, 1, 1, bias=False),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(256))
        self.rnns = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True)
        self.linears = nn.Sequential(
                        nn.Linear(hidden_size * 2, 1024),
                        nn.Dropout(p=0.2),
                        nn.Linear(1024, out_vocab))

    def forward(self, x, lengths):
        x = self.cnns(x.permute(0, 2, 1))
        x_packed = pack_padded_sequence(x.permute(2, 0, 1), lengths, enforce_sorted=False)
        out_packed = self.rnns(x_packed)[0]
        out, lens = pad_packed_sequence(out_packed)
        out = self.linears(out).log_softmax(2)
        return out, lens
