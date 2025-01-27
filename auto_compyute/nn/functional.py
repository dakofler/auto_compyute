"""Neural network functions"""

from typing import Optional

from ..autograd import Tensor, apply_func
from .funcs.activation_funcs import Softmax


def mse_loss(logits: Tensor, targets: Tensor):
    return ((logits - targets) ** 2).mean()


def linear(x: Tensor, w: Tensor, b: Optional[Tensor]) -> Tensor:
    y = x @ w.T
    if b is not None:
        return y + b
    return y


def relu(x: Tensor) -> Tensor:
    return x.maximum(0.0)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return apply_func(Softmax, x, dim=dim)


def tanh(x: Tensor) -> Tensor:
    return x.tanh()
