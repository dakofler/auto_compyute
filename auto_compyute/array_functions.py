"""Array functions"""

from .array_factory import array
from .autograd import Array, apply_func
from .backends import Scalar
from .dtypes import float32
from .funcs.shape_funcs import Concat, Stack, Where

__all__ = ["concat", "stack", "where"]


def concat(*arrays: Array, dim: int = 0):
    return apply_func(Concat, *arrays, dim=dim)


def stack(*arrays: Array, dim: int = 0):
    return apply_func(Stack, *arrays, dim=dim)


def where(condition: Array, x1: Array | Scalar, x2: Array | Scalar) -> Array:
    x1 = x1 if isinstance(x1, Array) else array(x1, condition.device, dtype=float32)
    x2 = x2 if isinstance(x2, Array) else array(x2, condition.device, dtype=float32)
    return apply_func(Where, condition, x1, x2)
