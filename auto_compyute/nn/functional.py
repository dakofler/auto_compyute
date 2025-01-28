"""Neural network functions"""

from typing import Optional

from ..autograd import Tensor, apply_func
from .funcs import Conv2D, Dilate2D, Linear, Maxpool2D, MSELoss, Pad2D, Softmax

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


def relu(x: Tensor) -> Tensor:
    return x.maximum(0.0)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return apply_func(Softmax, x, dim=dim)


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


def linear(x: Tensor, w: Tensor, b: Optional[Tensor]) -> Tensor:
    b = b if b is not None else x.self_like(0)
    return apply_func(Linear, x, w, b)


# -------------------------------------------------------------------------------------
# CONVOLUTION FUNCTIONS
# -------------------------------------------------------------------------------------


def conv2d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    if dilation > 1:
        w = apply_func(Dilate2D, w, dilation=dilation)
    if padding > 0:
        x = apply_func(Pad2D, x, padding=padding)
    y = apply_func(Conv2D, x, w, stride=stride)
    if b is not None:
        return y + b.view((b.shape[0], 1, 1))
    return y


def maxpool2d(x: Tensor, window_size: int) -> Tensor:
    return apply_func(Maxpool2D, x, window_size=window_size)


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


def mse_loss(logits: Tensor, targets: Tensor):
    return apply_func(MSELoss, logits, targets)
