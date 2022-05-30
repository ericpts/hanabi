import numpy as np
from typing import List
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpsilonGreedy(object):
    def __init__(self):
        self.start = 0.2
        self.end = 0.05
        self.decay = 1_000
        self.epoch = 0

    def get(self) -> float:
        decay_exponent = self.epoch / self.decay
        eps = self.end + (self.start - self.end) * np.math.exp(-decay_exponent)
        return eps

    def flip(self) -> bool:
        return random.random() < self.get()

    def on_epoch_start(self):
        self.epoch += 1


class ReplayBuffer(object):
    def __init__(self, max_size: int = 1_000):
        self.store = []
        self.at = 0
        self.max_size = max_size

    def __contains__(self, x):
        return x in self.store

    def push(self, x):
        if len(self.store) == self.max_size:
            self.store[self.at] = x
            self.at += 1
            if self.at == self.max_size:
                self.at = 0
        else:
            self.store.append(x)

    def sample(self, batch_size: int):
        return random.choices(self.store, k=batch_size)

    def __len__(self):
        return len(self.store)


class SimpleModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, fc_sizes: List[int]):
        super(SimpleModel, self).__init__()

        self.fcs = []
        last_size = input_size
        for s in fc_sizes:
            self.fcs.append(nn.Linear(last_size, s))
            last_size = s

        self.fcs = nn.ModuleList(self.fcs)

        self.last_fc = nn.Linear(last_size, output_size, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for f in self.fcs:
            X = f(X)
            X = F.relu(X)
        X = self.last_fc(X)
        return X
