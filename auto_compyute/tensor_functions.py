"""Tensor functions"""

from .autograd import Tensor, apply_func
from .backends import Scalar
from .dtypes import float32
from .funcs.shape_funcs import Concat, Stack, Where
from .tensor_factory import tensor

__all__ = ["concat", "stack", "where"]


def concat(*tensors: Tensor, dim: int = 0):
    return apply_func(Concat, *tensors, dim=dim)


def stack(*tensors: Tensor, dim: int = 0):
    return apply_func(Stack, *tensors, dim=dim)


def where(condition: Tensor, x1: Tensor | Scalar, x2: Tensor | Scalar) -> Tensor:
    x1 = x1 if isinstance(x1, Tensor) else tensor(x1, condition.device, dtype=float32)
    x2 = x2 if isinstance(x2, Tensor) else tensor(x2, condition.device, dtype=float32)
    return apply_func(Where, condition, x1, x2)
