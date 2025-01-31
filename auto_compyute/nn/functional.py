"""Neural network functions"""

import math
from typing import Optional

from ..autograd import Tensor, apply_func
from .funcs import (
    Batchnorm1D,
    Batchnorm2D,
    Conv2D,
    Dilate2D,
    Dropout,
    Linear,
    Maxpool2D,
    MSELoss,
    Pad2D,
    Softmax,
)

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
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tensor:
    if padding > 0:
        x = apply_func(Pad2D, x, padding=padding)
    if dilation > 1:
        w = apply_func(Dilate2D, w, dilation=dilation)
    y = apply_func(Conv2D, x, w, stride=stride)
    if b is not None:
        y += b.view((*b.shape, 1, 1))
    return y


def maxpool2d(x: Tensor, window_size: int = 2) -> Tensor:
    return apply_func(Maxpool2D, x, window_size=window_size)


# -------------------------------------------------------------------------------------
# ATTENTION FUNCTIONS
# -------------------------------------------------------------------------------------


def sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    *_, S, H = q.shape
    attn = q @ k.T / math.sqrt(H)
    if mask is not None:
        attn = attn + mask[:S, :S]
    attn = softmax(attn)
    return attn @ v


# -------------------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


def batchnorm1d(
    x: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    w: Tensor,
    b: Tensor,
    m: float = 0.1,
    eps: float = 1e-5,
    training: bool = False,
):
    return apply_func(
        Batchnorm1D, x, rmean, rvar, w, b, m=m, eps=eps, training=training
    )


def batchnorm2d(
    x: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    w: Tensor,
    b: Tensor,
    m: float = 0.1,
    eps: float = 1e-5,
    training: bool = False,
):
    return apply_func(
        Batchnorm2D, x, rmean, rvar, w, b, m=m, eps=eps, training=training
    )


# -------------------------------------------------------------------------------------
# REGULARIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0:
        return x
    return apply_func(Dropout, x, p=p)


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


def mse_loss(logits: Tensor, targets: Tensor):
    return apply_func(MSELoss, logits, targets)
