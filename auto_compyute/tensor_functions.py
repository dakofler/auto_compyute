"""Tensor functions."""

from .tensor_factory import tensor
from .autograd import Tensor, apply_op
from .backends import Scalar
from .dtypes import float32
from .ops.movement_ops import Concat, Stack, Where

__all__ = ["concat", "stack", "where"]


def concat(*tensors: Tensor, dim: int = 0):
    """Concatenates multiple tensors along a specified dimension.

    Args:
        *tensors (Tensor): Input tensors to concatenate.
        dim (int, optional): The dimension along which to concatenate. Defaults to `0`.

    Returns:
        Tensor: The concatenated result.
    """
    return apply_op(Concat, *tensors, dim=dim)


def stack(*tensors: Tensor, dim: int = 0):
    """Stacks multiple tensors along a new dimension.

    Args:
        *tensors (Tensor): Input tensors to stack.
        dim (int, optional): The dimension along which to stack the tensors. Defaults to `0`.

    Returns:
        Tensor: The stacked tensor with an added dimension.
    """
    return apply_op(Stack, *tensors, dim=dim)


def where(condition: Tensor, x1: Tensor | Scalar, x2: Tensor | Scalar) -> Tensor:
    """Selects elements from two tensors based on a condition.

    Args:
        condition (Tensor): A boolean tensor where `True` selects elements from `x1`, and
            `False` selects from `x2`.
        x1 (Tensor | Scalar): Values to select when `condition` is `True`.
        x2 (Tensor | Scalar): Values to select when `condition` is `False`.

    Returns:
        Tensor: An tensor with elements chosen from `x1` or `x2` based on `condition`.
    """
    x1 = x1 if isinstance(x1, Tensor) else tensor(x1, condition.device, dtype=float32)
    x2 = x2 if isinstance(x2, Tensor) else tensor(x2, condition.device, dtype=float32)
    return apply_op(Where, condition, x1, x2)
