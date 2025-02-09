"""Neural network functions."""

import math
from typing import Literal, Optional

from ..autograd import Array, _parse_key, apply_func
from ..dtypes import is_int
from . import funcs as NNFuncs

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


def gelu(x: Array) -> Array:
    """Applies the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (Array): Input tensor.

    Returns:
        Array: Output after applying GELU.
    """
    return apply_func(NNFuncs.GELU, x)


def relu(x: Array) -> Array:
    """Applies the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (Array): Input tensor.

    Returns:
        Array: Output after applying ReLU.
    """
    return apply_func(NNFuncs.ReLU, x)


def leaky_relu(x: Array, alpha: float = 0.2) -> Array:
    """Applies the Leaky ReLU activation function.

    Args:
        x (Array): Input tensor.
        alpha (float, optional): Slope for negative values. Defaults to `0.2`.

    Returns:
        Array: Output after applying Leaky ReLU.
    """
    return apply_func(NNFuncs.LeakyReLU, x, alpha=alpha)


def sigmoid(x: Array) -> Array:
    """Applies the sigmoid activation function.

    Args:
        x (Array): Input tensor.

    Returns:
        Array: Output after applying sigmoid.
    """
    return apply_func(NNFuncs.Sigmoid, x)


def softmax(x: Array, *, dim: int = -1) -> Array:
    """Applies the softmax activation function along a specified dimension.

    Args:
        x (Array): Input tensor.
        dim (int, optional): Dimension along which softmax is computed. Defaults to `-1`.

    Returns:
        Array: Output after applying softmax.
    """
    return apply_func(NNFuncs.Softmax, x, dim=dim)


def tanh(x: Array) -> Array:
    """Applies the hyperbolic tangent (tanh) activation function.

    Args:
        x (Array): Input tensor.

    Returns:
        Array: Output after applying tanh.
    """
    return x.tanh()


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


def linear(x: Array, w: Array, b: Optional[Array] = None) -> Array:
    """Applies a linear transformation.

    Args:
        x (Array): Input tensor.
        w (Array): Weight matrix.
        b (Array | None, optional): Bias vector. Defaults to `None`.

    Returns:
        Array: Output after applying the linear transformation.
    """
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
    """Applies a 2D convolution operation.

    Args:
        x (Array): Input tensor.
        w (Array): Kernel weight tensor.
        b (Array | None, optional): Bias vector. Defaults to `None`.
        stride (int, optional): Stride of the convolution. Defaults to `1`.
        padding (int, optional): Zero-padding added to all sides. Defaults to `0`.
        dilation (int, optional): Dilation factor for the kernel. Defaults to `1`.

    Returns:
        Array: Output after applying the 2D convolution.
    """
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
    output_padding: int = 0,
    dilation: int = 1,
) -> Array:
    """Applies a 2D transposed convolution operation.

    Args:
        x (Array): Input tensor.
        w (Array): Kernel weight tensor.
        b (Array | None, optional): Bias vector. Defaults to `None`.
        stride (int, optional): Stride of the transposed convolution. Defaults to `1`.
        padding (int, optional): Zero-padding added to both sides. Defaults to `0`.
        output_padding (int, optional): Additional size added to the output shape. Defaults to `0`.
        dilation (int, optional): Dilation factor for the kernel. Defaults to `1`.

    Returns:
        Array: Output after applying the 2D transposed convolution.
    """
    assert output_padding <= padding, "Output padding must be <= padding."
    if dilation > 1:
        w = apply_func(NNFuncs.Dilate2D, w, dilation=dilation)
    y = apply_func(NNFuncs.ConvTranspose2D, x, w, stride=stride)
    if padding > 0:
        y = apply_func(
            NNFuncs.OutPad2D, y, padding=padding, output_padding=output_padding
        )
    if b is not None:
        y += b.view(*b.shape, 1, 1)
    return y


def maxpool2d(x: Array, window_size: int = 2) -> Array:
    """Applies a 2D max pooling operation.

    Args:
        x (Array): Input tensor.
        window_size (int, optional): Size of the pooling window. Defaults to `2`.

    Returns:
        Array: Output after applying max pooling.
    """
    return apply_func(NNFuncs.Maxpool2D, x, window_size=window_size)


# -------------------------------------------------------------------------------------
# ATTENTION FUNCTIONS
# -------------------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: Array, k: Array, v: Array, mask: Optional[Array] = None, dropout_p: float = 0
) -> Array:
    """Computes scaled dot-product attention.

    Args:
        q (Array): Query array.
        k (Array): Key array.
        v (Array): Value array.
        mask (Array | None, optional): Attention mask. Defaults to `None`.
        dropout_p (float, optional): Dropout probability. Defaults to `0`.

    Returns:
        Array: Output after applying scaled dot-product attention.
    """
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
) -> Array:
    """Applies batch normalization.

    Args:
        x (Array): Input tensor.
        w (Array): Scale parameter (gamma).
        b (Array): Shift parameter (beta).
        rmean (Array): Running mean.
        rvar (Array): Running variance.
        momentum (float, optional): Momentum for updating running stats. Defaults to `0.1`.
        eps (float, optional): Small constant added for numerical stability. Defaults to `1e-5`.
        training (bool, optional): Whether to run in training mode. Defaults to `False`.
            If `training == True`, the running statistics are updated inplace.

    Notes:
        If `training == True`, the running statistics are updated inplace.

    Returns:
        Array: Output after applying batch normalization.
    """
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
    """Applies layer normalization.

    Args:
        x (Array): Input tensor.
        w (Array): Scale parameter (gamma).
        b (Array): Shift parameter (beta).
        eps (float, optional): Small constant added for numerical stability. Defaults to `1e-5`.

    Returns:
        Array: Output after applying layer normalization.
    """
    return apply_func(NNFuncs.Layernorm, x, w, b, eps=eps)


# -------------------------------------------------------------------------------------
# REGULARIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


def dropout(x: Array, p: float = 0.5, training: bool = True) -> Array:
    """Applies dropout regularization.

    Args:
        x (Array): Input tensor.
        p (float, optional): Dropout probability. Defaults to `0.5`.
        training (bool, optional): Whether the layer is in training mode. Defaults to `True`.

    Notes:
        Dropout is only applied during training. If `training == False` or `p == 0`, the input
            is returned unchanged.

    Returns:
        Array: Output after applying dropout.
    """
    if not training or p == 0:
        return x
    return apply_func(NNFuncs.Dropout, x, p=p)


# -------------------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
# -------------------------------------------------------------------------------------


def embedding(x: Array, emb_table: Array) -> Array:
    """Fetches embeddings for given input indices.

    Args:
        x (Array): Input indices array.
        emb_table (Array): Embedding table.

    Notes:
        The input `x` must contain integer values representing indices in the embedding table.

    Returns:
        Array: Output embeddings corresponding to input indices.
    """
    assert is_int(x.dtype)
    key = _parse_key(x)
    return apply_func(NNFuncs.Embedding, emb_table, key=key)


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


def mse_loss(logits: Array, targets: Array, reduction: Literal["sum", "mean"] = "mean"):
    """Computes Mean Squared Error (MSE) loss.

    Args:
        logits (Array): Predicted values.
        targets (Array): Ground truth values.
        reduction (Literal["sum", "mean"], optional): Specifies the reduction method.
            Defaults to "mean".

    Returns:
        Array: Computed MSE loss.
    """
    return apply_func(NNFuncs.MSELoss, logits, targets, reduction=reduction)


def cross_entropy_loss(
    logits: Array,
    targets: Array,
    eta: float = 1e-8,
    reduction: Literal["sum", "mean"] = "mean",
):
    """Computes Cross-Entropy loss.

    Args:
        logits (Array): Predicted logits.
        targets (Array): Ground truth labels.
        eta (float, optional): Small constant added for numerical stability. Defaults to `1e-8`.
        reduction (Literal["sum", "mean"], optional): Specifies the reduction method.
            Defaults to "mean".

    Notes:
        The `targets` must contain integer values representing class labels.

    Returns:
        Array: Computed cross-entropy loss.
    """
    assert is_int(targets.dtype)
    return apply_func(
        NNFuncs.CrossEntropyLoss, logits, targets, eta=eta, reduction=reduction
    )


def bce_loss(logits: Array, targets: Array, reduction: Literal["sum", "mean"] = "mean"):
    """Computes Binary Cross-Entropy (BCE) loss.

    Args:
        logits (Array): Predicted logits.
        targets (Array): Ground truth binary labels.
        reduction (Literal["sum", "mean"], optional): Specifies the reduction method.
            Defaults to "mean".

    Returns:
        Array: Computed BCE loss.
    """
    return apply_func(NNFuncs.BCELoss, logits, targets, reduction=reduction)
