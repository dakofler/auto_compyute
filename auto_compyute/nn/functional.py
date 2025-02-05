"""Neural network functions"""

import math
from typing import Literal, Optional

from ..autograd import Array, _parse_key, apply_func
from ..dtypes import is_int
from . import funcs as NNFuncs

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


def gelu(x: Array) -> Array:
    return apply_func(NNFuncs.GELU, x)


def relu(x: Array) -> Array:
    return apply_func(NNFuncs.ReLU, x)


def leaky_relu(x: Array, alpha: float = 0.2) -> Array:
    return apply_func(NNFuncs.LeakyReLU, x, alpha=alpha)


def sigmoid(x: Array) -> Array:
    return apply_func(NNFuncs.Sigmoid, x)


def softmax(x: Array, *, dim: int = -1) -> Array:
    return apply_func(NNFuncs.Softmax, x, dim=dim)


def tanh(x: Array) -> Array:
    return x.tanh()


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


def linear(x: Array, w: Array, b: Optional[Array]) -> Array:
    return apply_func(NNFuncs.Linear, x, w, b)


# -------------------------------------------------------------------------------------
# CONVOLUTION FUNCTIONS
# -------------------------------------------------------------------------------------


def conv2d(
    x: Array,
    w: Array,
    b: Optional[Array] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Array:
    if padding > 0:
        x = apply_func(NNFuncs.Pad2D, x, padding=padding)
    if dilation > 1:
        w = apply_func(NNFuncs.Dilate2D, w, dilation=dilation)
    y = apply_func(NNFuncs.Conv2D, x, w, stride=stride)
    if b is not None:
        y += b.view(*b.shape, 1, 1)
    return y


def conv_transpose2d(
    x: Array,
    w: Array,
    b: Optional[Array] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Array:
    if dilation > 1:
        w = apply_func(NNFuncs.Dilate2D, w, dilation=dilation)
    y = apply_func(NNFuncs.ConvTranspose2D, x, w, stride=stride)
    if padding > 0:
        y = apply_func(NNFuncs.InvPad2D, y, padding=padding)
    if b is not None:
        y += b.view(*b.shape, 1, 1)
    return y


def maxpool2d(x: Array, window_size: int = 2) -> Array:
    return apply_func(NNFuncs.Maxpool2D, x, window_size=window_size)


# -------------------------------------------------------------------------------------
# ATTENTION FUNCTIONS
# -------------------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: Array, k: Array, v: Array, mask: Optional[Array] = None, dropout_p: float = 0
) -> Array:
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
    x: Array,
    w: Array,
    b: Array,
    rmean: Array,
    rvar: Array,
    momentum: float = 0.1,
    eps: float = 1e-5,
    training: bool = False,
):
    return apply_func(
        NNFuncs.Batchnorm,
        x,
        rmean,
        rvar,
        w,
        b,
        momentum=momentum,
        eps=eps,
        training=training,
    )


def layernorm(x: Array, w: Array, b: Array, eps: float = 1e-5) -> Array:
    return apply_func(NNFuncs.Layernorm, x, w, b, eps=eps)


# -------------------------------------------------------------------------------------
# REGULARIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


def dropout(x: Array, p: float = 0.5, training: bool = True) -> Array:
    if not training or p == 0:
        return x
    return apply_func(NNFuncs.Dropout, x, p=p)


# -------------------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
# -------------------------------------------------------------------------------------


def embedding(x: Array, emb_table: Array) -> Array:
    assert is_int(x.dtype)
    key = _parse_key(x)
    return apply_func(NNFuncs.Embedding, emb_table, key=key)


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


def mse_loss(logits: Array, targets: Array, reduction: Literal["sum", "mean"] = "mean"):
    return apply_func(NNFuncs.MSELoss, logits, targets, reduction=reduction)


def cross_entropy_loss(
    logits: Array,
    targets: Array,
    eta: float = 1e-8,
    reduction: Literal["sum", "mean"] = "mean",
):
    return apply_func(
        NNFuncs.CrossEntropyLoss, logits, targets, eta=eta, reduction=reduction
    )


def bce_loss(logits: Array, targets: Array, reduction: Literal["sum", "mean"] = "mean"):
    return apply_func(NNFuncs.BCELoss, logits, targets, reduction=reduction)
