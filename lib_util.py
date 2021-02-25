from contextlib import contextmanager
import string
import time
import numpy as np
import torch


@contextmanager
def timeit(message: str):
    t0 = time.time()
    try:
        yield None
    finally:
        t1 = time.time()
        elapsed = t1 - t0
        print(f"Spent {elapsed * 1_000:.0f}ms in {message}.")


def as_torch(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x).float()


class CompactLogger(object):
    def __init__(self, log_every_n: int = 100):
        self.n_lines = 0
        self.log_every_n = log_every_n
        self.at = 0

    def on_episode_start(self):
        self.at += 1
        if self.at == self.log_every_n:
            self.at = 0
            self.reset()

    def print(self, message: str):
        if self.at != 0:
            return
        self.n_lines += 1
        print(message)

    def reset(self):
        for _ in range(self.n_lines):
            print("\033[F" + (" " * 80), end="")
        print("\r", end="")
        self.n_lines = 0


compact_logger = CompactLogger()


def card_to_str(card):
    alphabet = string.ascii_uppercase
    (suit_index, number) = card
    if suit_index < len(alphabet):
        suit = alphabet[suit_index]
    else:
        suit = f"({suit_index})"
    return f"{suit}{number}"
