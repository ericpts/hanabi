import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib_types import GameConfig


class EpsilonGreedy(object):
    def __init__(self):
        self.start = 0.9
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
    def __init__(self, max_size: int = 200_000):
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
    def __init__(self, game_config: GameConfig):
        super(SimpleModel, self).__init__()
        n_players = game_config.n_players
        hand_size = game_config.hand_size
        n_suits = game_config.n_suits
        n_ranks = game_config.n_ranks
        max_lives = game_config.max_lives
        input_size = (
            # Encoding each player's hand.
            n_players * hand_size * (n_suits + n_ranks)
            # Stacks.
            + n_suits * (n_ranks + 1)
            # Remaining cards in the game.
            + n_suits * n_ranks
            # Number of lives left.
            + (max_lives + 1)
        )

        print(f"Input size is {input_size}.")

        action_space = 2 * hand_size

        self.game_config = game_config
        self.hand_size = hand_size
        self.fc_sizes = [100, 100]

        self.fcs = []
        last_size = input_size
        for s in self.fc_sizes:
            self.fcs.append(nn.Linear(last_size, s, bias=False))
            last_size = s

        self.last_fc = nn.Linear(self.fc_sizes[-1], action_space, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for f in self.fcs:
            X = f(X)
            X = F.relu(X)
        X = self.last_fc(X)
        return X
