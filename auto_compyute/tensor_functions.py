"""Tensor functions"""

from .autograd import Tensor, apply_func, as_tensor
from .backends import Scalar
from .funcs.shape_funcs import Concat, Stack, Where

__all__ = ["concat", "stack", "where"]


def concat(*tensors: Tensor, dim: int = 0):
    return apply_func(Concat, *tensors, dim=dim)


def stack(*tensors: Tensor, dim: int = 0):
    return apply_func(Stack, *tensors, dim=dim)


def where(condition: Tensor, x1: Tensor | Scalar, x2: Tensor | Scalar) -> Tensor:
    device = condition.device
    x1, x2 = as_tensor(x1, device), as_tensor(x2, device)
    return apply_func(Where, condition, x1, x2)
