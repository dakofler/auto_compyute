"""Tensor functions"""

from .autograd import Tensor, apply_func
from .backends import Scalar
from .funcs.shape_funcs import Concat, Stack, Where

__all__ = ["concat", "stack", "where"]


def concat(*tensors: Tensor, dim: int = 0):
    return apply_func(Concat, *tensors, dim=dim)


def stack(*tensors: Tensor, dim: int = 0):
    return apply_func(Stack, *tensors, dim=dim)


def where(condition: Tensor, x1: Tensor | Scalar, x2: Tensor | Scalar) -> Tensor:
    x1, x2 = condition.self_like(x1), condition.self_like(x2)
    return apply_func(Where, condition, x1, x2)
