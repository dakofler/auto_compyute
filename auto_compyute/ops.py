from typing import Any

from .autograd import Node, apply_function
from .backends import Array
from .functions import Maximum

__all__ = ["maximum"]


def maximum(x1: Array, x2: Any) -> Node:
    return apply_function(Maximum, x1, x2)
