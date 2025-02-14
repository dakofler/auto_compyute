"""Neural network functions."""

from typing import Literal, Optional

from ..autograd import Tensor, _parse_key, apply_op
from ..dtypes import is_int
from . import ops as NNFuncs

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


def gelu(x: Tensor) -> Tensor:
    """Applies the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output after applying GELU.
    """
    return apply_op(NNFuncs.GELU, x)


def relu(x: Tensor) -> Tensor:
    """Applies the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output after applying ReLU.
    """
    return apply_op(NNFuncs.ReLU, x)


def leaky_relu(x: Tensor, alpha: float = 0.2) -> Tensor:
    """Applies the Leaky ReLU activation function.

    Args:
        x (Tensor): Input tensor.
        alpha (float, optional): Slope for negative values. Defaults to `0.2`.

    Returns:
        Tensor: Output after applying Leaky ReLU.
    """
    return apply_op(NNFuncs.LeakyReLU, x, alpha=alpha)


def sigmoid(x: Tensor) -> Tensor:
    """Applies the sigmoid activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output after applying sigmoid.
    """
    return apply_op(NNFuncs.Sigmoid, x)


def softmax(x: Tensor, *, dim: int = -1) -> Tensor:
    """Applies the softmax activation function along a specified dimension.

    Args:
        x (Tensor): Input tensor.
        dim (int, optional): Dimension along which softmax is computed. Defaults to `-1`.

    Returns:
        Tensor: Output after applying softmax.
    """
    return apply_op(NNFuncs.Softmax, x, dim=dim)


def tanh(x: Tensor) -> Tensor:
    """Applies the hyperbolic tangent (tanh) activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output after applying tanh.
    """
    return x.tanh()


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Applies a linear transformation.

    Args:
        x (Tensor): Input tensor.
        w (Tensor): Weight matrix.
        b (Tensor | None, optional): Bias vector. Defaults to `None`.

    Returns:
        Tensor: Output after applying the linear transformation.
    """
    return apply_op(NNFuncs.Linear, x, w, b)


# -------------------------------------------------------------------------------------
# CONVOLUTION FUNCTIONS
# -------------------------------------------------------------------------------------


def conv1d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tensor:
    """Applies a 1D convolution operation.

    Args:
        x (Tensor): Input tensor.
        w (Tensor): Kernel weight tensor.
        b (Tensor | None, optional): Bias vector. Defaults to `None`.
        stride (int, optional): Stride of the convolution. Defaults to `1`.
        padding (int, optional): Zero-padding added to all sides. Defaults to `0`.
        dilation (int, optional): Dilation factor for the kernel. Defaults to `1`.

    Returns:
        Tensor: Output after applying the 1D convolution.
    """
    if padding > 0:
        x = apply_op(NNFuncs.Pad1D, x, padding=padding)
    if dilation > 1:
        w = apply_op(NNFuncs.Dilate1D, w, dilation=dilation)
    y = apply_op(NNFuncs.Conv1D, x, w, stride=stride)
    if b is not None:
        y += b.view(*b.shape, 1)
    return y


def conv2d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tensor:
    """Applies a 2D convolution operation.

    Args:
        x (Tensor): Input tensor.
        w (Tensor): Kernel weight tensor.
        b (Tensor | None, optional): Bias vector. Defaults to `None`.
        stride (int, optional): Stride of the convolution. Defaults to `1`.
        padding (int, optional): Zero-padding added to all sides. Defaults to `0`.
        dilation (int, optional): Dilation factor for the kernel. Defaults to `1`.

    Returns:
        Tensor: Output after applying the 2D convolution.
    """
    if padding > 0:
        x = apply_op(NNFuncs.Pad2D, x, padding=padding)
    if dilation > 1:
        w = apply_op(NNFuncs.Dilate2D, w, dilation=dilation)
    y = apply_op(NNFuncs.Conv2D, x, w, stride=stride)
    if b is not None:
        y += b.view(*b.shape, 1, 1)
    return y


def conv_transpose2d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
) -> Tensor:
    """Applies a 2D transposed convolution operation.

    Args:
        x (Tensor): Input tensor.
        w (Tensor): Kernel weight tensor.
        b (Tensor | None, optional): Bias vector. Defaults to `None`.
        stride (int, optional): Stride of the transposed convolution. Defaults to `1`.
        padding (int, optional): Zero-padding added to both sides. Defaults to `0`.
        output_padding (int, optional): Additional size added to the output shape. Defaults to `0`.
        dilation (int, optional): Dilation factor for the kernel. Defaults to `1`.

    Returns:
        Tensor: Output after applying the 2D transposed convolution.
    """
    assert output_padding <= padding, "Output padding must be <= padding."
    if dilation > 1:
        w = apply_op(NNFuncs.Dilate2D, w, dilation=dilation)
    y = apply_op(NNFuncs.ConvTranspose2D, x, w, stride=stride)
    if padding > 0:
        y = apply_op(
            NNFuncs.OutPad2D, y, padding=padding, output_padding=output_padding
        )
    if b is not None:
        y += b.view(*b.shape, 1, 1)
    return y


def maxpool2d(x: Tensor, window_size: int = 2) -> Tensor:
    """Applies a 2D max pooling operation.

    Args:
        x (Tensor): Input tensor.
        window_size (int, optional): Size of the pooling window. Defaults to `2`.

    Returns:
        Tensor: Output after applying max pooling.
    """
    return apply_op(NNFuncs.Maxpool2D, x, window_size=window_size)


# -------------------------------------------------------------------------------------
# ATTENTION FUNCTIONS
# -------------------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, dropout_p: float = 0
) -> Tensor:
    """Computes scaled dot-product attention.

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        v (Tensor): Value tensor.
        mask (Tensor | None, optional): Attention mask. Defaults to `None`.
        dropout_p (float, optional): Dropout probability. Defaults to `0`.

    Returns:
        Tensor: Output after applying scaled dot-product attention.
    """
    return apply_op(NNFuncs.ScaledDotProductAttention, q, k, v, mask, p=dropout_p)


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
) -> Tensor:
    """Applies batch normalization.

    Args:
        x (Tensor): Input tensor.
        w (Tensor): Scale parameter (gamma).
        b (Tensor): Shift parameter (beta).
        rmean (Tensor): Running mean.
        rvar (Tensor): Running variance.
        momentum (float, optional): Momentum for updating running stats. Defaults to `0.1`.
        eps (float, optional): Small constant added for numerical stability. Defaults to `1e-5`.
        training (bool, optional): Whether to run in training mode. Defaults to `False`.
            If `training == True`, the running statistics are updated inplace.

    Notes:
        If `training == True`, the running statistics are updated inplace.

    Returns:
        Tensor: Output after applying batch normalization.
    """
    return apply_op(
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


def layernorm(x: Tensor, w: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
    """Applies layer normalization.

    Args:
        x (Tensor): Input tensor.
        w (Tensor): Scale parameter (gamma).
        b (Tensor): Shift parameter (beta).
        eps (float, optional): Small constant added for numerical stability. Defaults to `1e-5`.

    Returns:
        Tensor: Output after applying layer normalization.
    """
    return apply_op(NNFuncs.Layernorm, x, w, b, eps=eps)


# -------------------------------------------------------------------------------------
# REGULARIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Applies dropout regularization.

    Args:
        x (Tensor): Input tensor.
        p (float, optional): Dropout probability. Defaults to `0.5`.
        training (bool, optional): Whether the layer is in training mode. Defaults to `True`.

    Notes:
        Dropout is only applied during training. If `training == False` or `p == 0`, the input
            is returned unchanged.

    Returns:
        Tensor: Output after applying dropout.
    """
    if not training or p == 0:
        return x
    return apply_op(NNFuncs.Dropout, x, p=p)


# -------------------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
# -------------------------------------------------------------------------------------


def embedding(x: Tensor, emb_table: Tensor) -> Tensor:
    """Fetches embeddings for given input indices.

    Args:
        x (Tensor): Input indices tensor.
        emb_table (Tensor): Embedding table.

    Notes:
        The input `x` must contain integer values representing indices in the embedding table.

    Returns:
        Tensor: Output embeddings corresponding to input indices.
    """
    assert is_int(x.dtype)
    key = _parse_key(x)
    return apply_op(NNFuncs.Embedding, emb_table, key=key)


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


def mse_loss(
    logits: Tensor, targets: Tensor, reduction: Literal["sum", "mean"] = "mean"
):
    """Computes Mean Squared Error (MSE) loss.

    Args:
        logits (Tensor): Predicted values.
        targets (Tensor): Ground truth values.
        reduction (Literal["sum", "mean"], optional): Specifies the reduction method.
            Defaults to "mean".

    Returns:
        Tensor: Computed MSE loss.
    """
    return apply_op(NNFuncs.MSELoss, logits, targets, reduction=reduction)


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    eta: float = 1e-8,
    reduction: Literal["sum", "mean"] = "mean",
):
    """Computes Cross-Entropy loss.

    Args:
        logits (Tensor): Predicted logits.
        targets (Tensor): Ground truth labels.
        eta (float, optional): Small constant added for numerical stability. Defaults to `1e-8`.
        reduction (Literal["sum", "mean"], optional): Specifies the reduction method.
            Defaults to "mean".

    Notes:
        The `targets` must contain integer values representing class labels.

    Returns:
        Tensor: Computed cross-entropy loss.
    """
    assert is_int(targets.dtype)
    return apply_op(
        NNFuncs.CrossEntropyLoss, logits, targets, eta=eta, reduction=reduction
    )


def bce_loss(
    logits: Tensor, targets: Tensor, reduction: Literal["sum", "mean"] = "mean"
):
    """Computes Binary Cross-Entropy (BCE) loss.

    Args:
        logits (Tensor): Predicted logits.
        targets (Tensor): Ground truth binary labels.
        reduction (Literal["sum", "mean"], optional): Specifies the reduction method.
            Defaults to "mean".

    Returns:
        Tensor: Computed BCE loss.
    """
    return apply_op(NNFuncs.BCELoss, logits, targets, reduction=reduction)
