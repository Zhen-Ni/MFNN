#!/usr/bin/env python3

from __future__ import annotations

import gc
import sys
import typing
from io import BytesIO
import typing

import torch

# __all__ = ('DEVICE', 'free_memory', 'copy_to', 'Statistics', )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def copy_to(data, device: torch.device | int | str | None = None):
    """Copy data to device."""
    # Avoid useless copy in gpu.
    # See https://discuss.pytorch.org/t/how-to-make-a-copy-of-a-gpu-model-on-the-cpu/90955/4
    if device is None:
        return data
    memory = BytesIO()
    torch.save(data, memory, pickle_protocol=-1)
    memory.seek(0)
    data = torch.load(memory, map_location=device)
    memory.close()
    return data


T = typing.TypeVar('T', bound=float)


class Statistics(typing.Generic[T]):
    """Get statistical value(sum, average, variance) of data."""
    __slots__ = '_count', '_value', '_s1', '_s2'

    def __init__(self):
        self._count = 0
        self._value: T = 0.
        self._s1: float = 0.            # sum of sample values
        self._s2: float = 0.            # sum of squared sample values

    def update(self, value: T, n: int = 1) -> Statistics[T]:
        self._count += n
        self._value = value
        self._s1 += self.value * n
        self._s2 += value ** 2 * n
        return self

    @property
    def count(self) -> int:
        return self._count

    @property
    def value(self) -> T:
        return self._value

    @property
    def sum(self) -> float:
        return self._s1

    @property
    def average(self) -> float:
        return self._s1 / self._count

    @property
    def variance(self) -> float:
        return self._s2 / self._count - self.average ** 2

    @property
    def std(self) -> float:
        return self.variance ** .5


class GatedStdout:
    def __init__(self, suppress: bool):
        self.suppress = suppress

    def write(self, s: typing.TypeVar('AnyStr', bytes, str)):
        if self.suppress:
            return 0
        return sys.stdout.write(s)
