"""Array backends"""

from types import ModuleType
from typing import Optional, TypeAlias

import cupy as cp
import numpy as np

__all__ = ["cpu", "cuda"]

Array: TypeAlias = cp.ndarray | np.ndarray
Scalar: TypeAlias = int | float
Shape = tuple[int, ...]
Dim = int | tuple[int, ...]


class Backend:
    def __init__(self, module: ModuleType):
        self.m = module

    def __repr__(self):
        return self.__class__.__name__


cpu = Backend(np)
cuda = Backend(cp)


def get_array_backend(x: Array) -> Backend:
    return cuda if isinstance(x, cp.ndarray) else cpu


def select_backend(device: Optional[Backend]) -> Backend:
    return device if device is not None else cpu
