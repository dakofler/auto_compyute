from typing import Any

from .backends import Array
from .functions import Maximum
from .node import Node, apply_function

__all__ = ["maximum"]


def maximum(x1: Array, x2: Any) -> Node:
    return apply_function(Maximum, x1, x2)
