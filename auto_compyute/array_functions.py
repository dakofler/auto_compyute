"""Array functions."""

from .array_factory import array
from .autograd import Array, apply_func
from .backends import Scalar
from .dtypes import float32
from .funcs.shape_funcs import Concat, Stack, Where

__all__ = ["concat", "stack", "where"]


def concat(*arrays: Array, dim: int = 0):
    """Concatenates multiple arrays along a specified dimension.

    Args:
        *arrays (Array): Input arrays to concatenate.
        dim (int, optional): The dimension along which to concatenate. Defaults to `0`.

    Returns:
        Array: The concatenated result.
    """
    return apply_func(Concat, *arrays, dim=dim)


def stack(*arrays: Array, dim: int = 0):
    """Stacks multiple arrays along a new dimension.

    Args:
        *arrays (Array): Input arrays to stack.
        dim (int, optional): The dimension along which to stack the arrays. Defaults to `0`.

    Returns:
        Array: The stacked array with an added dimension.
    """
    return apply_func(Stack, *arrays, dim=dim)


def where(condition: Array, x1: Array | Scalar, x2: Array | Scalar) -> Array:
    """Selects elements from two arrays based on a condition.

    Args:
        condition (Array): A boolean array where `True` selects elements from `x1`, and
            `False` selects from `x2`.
        x1 (Array | Scalar): Values to select when `condition` is `True`.
        x2 (Array | Scalar): Values to select when `condition` is `False`.

    Returns:
        Array: An array with elements chosen from `x1` or `x2` based on `condition`.
    """
    x1 = x1 if isinstance(x1, Array) else array(x1, condition.device, dtype=float32)
    x2 = x2 if isinstance(x2, Array) else array(x2, condition.device, dtype=float32)
    return apply_func(Where, condition, x1, x2)
