"""Neural network functions"""

import math
from typing import Literal, Optional

from ..autograd import Tensor, _parse_key, apply_func
from ..dtypes import is_int
from .funcs import (
    GELU,
    Batchnorm,
    BCELoss,
    Conv2D,
    CrossEntropyLoss,
    Dilate2D,
    Dropout,
    Embedding,
    Layernorm,
    LeakyReLU,
    Linear,
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


def leaky_relu(x: Tensor, alpha: float = 0.2) -> Tensor:
    return apply_func(LeakyReLU, x, alpha=alpha)


def sigmoid(x: Tensor) -> Tensor:
    return apply_func(Sigmoid, x)


def softmax(x: Tensor, *, dim: int = -1) -> Tensor:
    return apply_func(Softmax, x, dim=dim)


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


def linear(x: Tensor, w: Tensor, b: Optional[Tensor]) -> Tensor:
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
        y += b.view(*b.shape, 1, 1)
    return y


def maxpool2d(x: Tensor, window_size: int = 2) -> Tensor:
    return apply_func(Maxpool2D, x, window_size=window_size)


# -------------------------------------------------------------------------------------
# ATTENTION FUNCTIONS
# -------------------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, dropout_p: float = 0
) -> Tensor:
    *_, seq_len, head_size = q.shape

    attn = q @ k.T / math.sqrt(head_size)
    if mask is not None:
        attn += mask[:seq_len, :seq_len]
    attn = softmax(attn, dim=-1)
    attn = dropout(attn, dropout_p)
    return attn @ v


# -------------------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


def batchnorm(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    momentum: float = 0.1,
    eps: float = 1e-5,
    training: bool = False,
):
    return apply_func(
        Batchnorm, x, rmean, rvar, w, b, momentum=momentum, eps=eps, training=training
    )


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
# EMBEDDING FUNCTIONS
# -------------------------------------------------------------------------------------


def embedding(x: Tensor, emb_table: Tensor) -> Tensor:
    assert is_int(x.dtype)
    key = _parse_key(x)
    return apply_func(Embedding, emb_table, key=key)


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


def mse_loss(
    logits: Tensor, targets: Tensor, reduction: Literal["sum", "mean"] = "mean"
):
    return apply_func(MSELoss, logits, targets, reduction=reduction)


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    eta: float = 1e-8,
    reduction: Literal["sum", "mean"] = "mean",
):
    return apply_func(CrossEntropyLoss, logits, targets, eta=eta, reduction=reduction)


def bce_loss(
    logits: Tensor, targets: Tensor, reduction: Literal["sum", "mean"] = "mean"
):
    return apply_func(BCELoss, logits, targets, reduction=reduction)
