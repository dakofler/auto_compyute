"""Neural network functions"""

import math
from typing import Optional

from ..autograd import Tensor, apply_func
from ..dtypes import int64
from .funcs import (
    GELU,
    Conv2D,
    CrossEntropyLoss,
    Dilate2D,
    Dropout,
    Layernorm,
    Maxpool2D,
    MSELoss,
    Pad2D,
    ReLU,
    Sigmoid,
    Softmax,
)

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


def gelu(x: Tensor) -> Tensor:
    return apply_func(GELU, x)


def relu(x: Tensor) -> Tensor:
    return apply_func(ReLU, x)


def sigmoid(x: Tensor) -> Tensor:
    return apply_func(Sigmoid, x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return apply_func(Softmax, x, dim=dim)


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


def linear(x: Tensor, w: Tensor, b: Optional[Tensor]) -> Tensor:
    y = x @ w.T
    if b is not None:
        y += b
    return y


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


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, dropout_p: float = 0
) -> Tensor:
    *_, S, H = q.shape
    attn = q @ k.T / math.sqrt(H)
    if mask is not None:
        attn += mask[:S, :S]
    attn = softmax(attn)
    attn = dropout(attn, dropout_p, dropout_p > 0)
    return attn @ v


# -------------------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


def batchnorm(
    x: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    w: Tensor,
    b: Tensor,
    m: float = 0.1,
    eps: float = 1e-5,
    training: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_dims = (0,) + tuple(d for d in range(x.ndim) if d > 1)
    ext_shape = (1,) * (x.ndim - 2)

    if training:
        mean = x.mean(batch_dims, keepdims=True)
        std = (x.var(batch_dims, ddof=0, keepdims=True) + eps).sqrt()
        rmean = rmean * (1 - m) + mean.squeeze() * m
        rvar = rvar * (1 - m) + x.var(batch_dims) * m
    else:
        mean = rmean.view((*rmean.shape, *ext_shape))
        std = (rvar.view((*rvar.shape, *ext_shape)) + eps).sqrt()

    w = w.view((*w.shape, *ext_shape))
    b = b.view((*b.shape, *ext_shape))
    y = (x - mean) / std * w + b
    return y, rmean, rvar


def layernorm(x: Tensor, w: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
    return apply_func(Layernorm, x, w, b, eps=eps)


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


def cross_entropy(logits: Tensor, targets: Tensor, eta: float = 1e-8):
    assert targets.dtype == int64
    return apply_func(CrossEntropyLoss, logits, targets, eta=eta)
