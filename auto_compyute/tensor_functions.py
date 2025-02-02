"""Tensor functions"""

from .autograd import Tensor, apply_func
from .funcs.shape_funcs import Concat, Stack

__all__ = ["concat", "stack"]


def concat(*tensors: Tensor, dim: int = 0):
    return apply_func(Concat, *tensors, dim=dim)


def stack(*tensors: Tensor, dim: int = 0):
    return apply_func(Stack, *tensors, dim=dim)
