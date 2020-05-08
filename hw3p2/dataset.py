import numpy as np
import torch
from torch.nn.utils.rnn import *
from torch.utils.data import Dataset
from phoneme_list import *


class Phoneme(Dataset):
    def __init__(self, x, y=None, pred=False):
        self.pred = pred
        self.x = self.normalize(x)
        self.y = y
        self.x_lens, self.y_lens = self.pad()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if not self.pred:
            return self.x[i], self.y[i], self.x_lens[i], self.y_lens[i]
        else:
            return self.x[i], self.x_lens[i]

    def normalize(self, x):
        n = x.shape[0]
        for i in range(n):
            x[i] = (x[i] - np.mean(x[i], axis=0, keepdims=True)) / np.std(x[i], axis=0, keepdims=True)
        return x

    def pad(self):
        self.x = [torch.tensor(x) for x in self.x]
        x_lens = torch.LongTensor([x.shape[0] for x in self.x])
        self.x = pad_sequence(self.x, batch_first=True)
        if not self.pred:
            self.y = [torch.tensor(y) for y in self.y]
            y_lens = torch.LongTensor([y.shape[0] for y in self.y])
            self.y = pad_sequence(self.y, batch_first=True)
            return x_lens, y_lens
        else:
            return x_lens, None
