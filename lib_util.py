from contextlib import contextmanager
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


def as_torch(x: np.ndarray) -> torch.FloatTensor:
    return torch.tensor(x).float()
