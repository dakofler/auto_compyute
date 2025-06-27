"""Neural network differentiable operations."""

import math
from types import ModuleType
from typing import Optional

from opt_einsum import contract as einsum  # type: ignore

from auto_compyute.backends import Array, ShapeLike
from auto_compyute.ops.movement_ops import Select
from auto_compyute.ops.op import Op

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTION OPERATIONS
# -------------------------------------------------------------------------------------


class GELU(Op):
    """Gaussian error linear unit activation function."""

    def forward(self, x: Array) -> Array:
        y = 0.5 * x * (1 + self.xp.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
        self.stash(x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (x,) = self.unstash()
        t = self.xp.tanh(x * 0.79788 * (1 + 0.04472 * x * x))
        dx = dy * 0.5 * ((1 + t) + x * (1 - t * t) * (0.79788 + 0.10703 * x * x))
        return (dx,)


class ReLU(Op):
    """Rectified linear unit activation function."""

    def forward(self, x: Array) -> Array:
        y = self.xp.maximum(x, 0.0)
        self.stash(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (mask,) = self.unstash()
        dx = dy * mask
        return (dx,)


class LeakyReLU(Op):
    """Leaky rectified linear unit activation function."""

    def forward(self, x: Array, *, alpha: float) -> Array:
        y = self.xp.maximum(x, x * alpha)
        self.stash(alpha, y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        alpha, mask = self.unstash()
        dx = dy * (mask + self.xp.invert(mask).astype(dy.dtype) * alpha)
        return (dx,)


def _sigmoid_fwd(xp: ModuleType, x: Array) -> Array:
    return 1 / (1 + xp.exp(-x))


class Sigmoid(Op):
    """Sigmoid activation function."""

    def forward(self, x: Array) -> Array:
        y = _sigmoid_fwd(self.xp, x)
        self.stash(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (y,) = self.unstash()
        dx = dy * y * (1 - y)
        return (dx,)


def _softmax_fwd(xp: ModuleType, x: Array, dim: int) -> Array:
    x = xp.exp(x - x.max(dim, keepdims=True))
    return x / x.sum(dim, keepdims=True)


def _softmax_bwd(y: Array, dy: Array, dim: int) -> Array:
    return y * (dy - (dy * y).sum(dim, keepdims=True))


class Softmax(Op):
    """Softmax activation function."""

    def forward(self, x: Array, *, dim: int) -> Array:
        y = _softmax_fwd(self.xp, x, dim)
        self.stash(dim, y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, y = self.unstash()
        dx = _softmax_bwd(y, dy, dim)
        return (dx,)


# -------------------------------------------------------------------------------------
# LINEAR OPERATIONS
# -------------------------------------------------------------------------------------


class Linear(Op):
    """Linear projection."""

    def forward(self, x: Array, w: Array, b: Optional[Array]) -> Array:
        y = x @ w.swapaxes(-1, -2)  # x @ w.T
        y = y if b is None else y + b
        self.stash(x, w, b is not None)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, w, b_requires_grad = self.unstash()
        dx = dy @ w
        dw = dy.swapaxes(-1, -2) @ x  # dy.T @ x
        db = None if not b_requires_grad else dy
        return dx, dw, db


# -------------------------------------------------------------------------------------
# CONVOLUTION OPERATIONS
# -------------------------------------------------------------------------------------


def _pad1d_fwd(xp: ModuleType, x: Array, left_pad: int, right_pad: Optional[int] = None) -> Array:
    right_pad = right_pad if right_pad is not None else left_pad
    paddings = tuple([(0, 0)] * (x.ndim - 1) + [(left_pad, right_pad)])
    return xp.pad(x, paddings)


def _pad1d_bwd(dy: Array, left_pad: int, right_pad: Optional[int] = None) -> Array:
    right_pad = right_pad if right_pad is not None else left_pad
    if right_pad <= 0:
        return dy[..., left_pad:]
    return dy[..., left_pad:-right_pad]


class Pad1D(Op):
    """1D padding."""

    def forward(self, x: Array, *, padding: int) -> Array:
        y = _pad1d_fwd(self.xp, x, padding, padding)
        self.stash(padding)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (padding,) = self.unstash()
        dx = _pad1d_bwd(dy, padding, padding)
        return (dx,)


def _dilate1d_fwd(xp: ModuleType, x: Array, dilation: int) -> Array:
    y_size = dilation * (x.shape[-1] - 1) + 1
    y_shape = (*x.shape[:-1], y_size)
    y = xp.zeros(y_shape, dtype=x.dtype)
    y[..., ::dilation] = x
    return y


def _dilate1d_bwd(dy: Array, dilation: int) -> Array:
    return dy[..., ::dilation]


class Dilate1D(Op):
    """1D dilation."""

    def forward(self, x: Array, *, dilation: int) -> Array:
        y = _dilate1d_fwd(self.xp, x, dilation)
        self.stash(dilation)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (dilation,) = self.unstash()
        dx = _dilate1d_bwd(dy, dilation)
        return (dx,)


def _pad_to_shape(xp: ModuleType, x: Array, shape: ShapeLike) -> Array:
    padding = tuple((0, shape[i] - x.shape[i]) for i in range(x.ndim))
    return xp.pad(x, padding)


def _windowed_view_1d(xp: ModuleType, x: Array, window_size: int, stride: int = 1) -> Array:
    # compute strided shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-1], out, window_size)

    # compute strides
    xstr = x.strides
    out_strides = (*xstr[:-1], xstr[-1] * stride, xstr[-1])

    return xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)


class Conv1D(Op):
    """1D convolution."""

    def forward(self, x: Array, w: Array, *, stride: int) -> Array:
        kernel_size = w.shape[-1]

        # convolve
        x_windowed = _windowed_view_1d(self.xp, x, kernel_size, stride)
        y = einsum("bitk,oik->bot", x_windowed, w, use_blas=True)

        self.stash(x, w, x.shape[-1], kernel_size, stride)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, w, input_size, kernel_size, stride = self.unstash()

        # dilate and full pad
        if stride > 1:
            dy = _dilate1d_fwd(self.xp, dy, stride)
        output_size = input_size - kernel_size + 1
        output_shape = (*dy.shape[:-1], output_size)
        dy = _pad_to_shape(self.xp, dy, output_shape)
        dy = _pad1d_fwd(self.xp, dy, kernel_size - 1)

        # input grad
        w = self.xp.flip(w, axis=-1)
        dy_windowed = _windowed_view_1d(self.xp, dy, kernel_size)
        dx = einsum("botk,oik->bit", dy_windowed, w, use_blas=True)

        # weight grad
        dy_windowed = _windowed_view_1d(self.xp, dy, input_size)
        dw = einsum("bokt,bit->oik", dy_windowed, x, use_blas=True)
        dw = self.xp.flip(dw, axis=-1)

        return dx, dw


def _pad2d_fwd(xp: ModuleType, x: Array, left_pad: int, right_pad: Optional[int] = None) -> Array:
    right_pad = right_pad if right_pad is not None else left_pad
    paddings = tuple([(0, 0)] * (x.ndim - 2) + [(left_pad, right_pad)] * 2)
    return xp.pad(x, paddings)


def _pad2d_bwd(dy: Array, left_pad: int, right_pad: Optional[int] = None) -> Array:
    right_pad = right_pad if right_pad is not None else left_pad
    if right_pad <= 0:
        return dy[..., left_pad:, left_pad:]
    return dy[..., left_pad:-right_pad, left_pad:-right_pad]


class Pad2D(Op):
    """2D padding."""

    def forward(self, x: Array, *, padding: int) -> Array:
        y = _pad2d_fwd(self.xp, x, padding, padding)
        self.stash(padding)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (padding,) = self.unstash()
        dx = _pad2d_bwd(dy, padding, padding)
        return (dx,)


class OutPad2D(Op):
    """2D output padding."""

    def forward(self, x: Array, *, padding: int, output_padding: int) -> Array:
        y = _pad2d_bwd(x, padding, padding - output_padding)
        self.stash(padding, output_padding)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        padding, output_padding = self.unstash()
        dx = _pad2d_fwd(self.xp, dy, padding, padding - output_padding)
        return (dx,)


def _dilate2d_fwd(xp: ModuleType, x: Array, dilation: int) -> Array:
    y_size = dilation * (x.shape[-1] - 1) + 1
    y_shape = (*x.shape[:-2], y_size, y_size)
    y = xp.zeros(y_shape, dtype=x.dtype)
    y[..., ::dilation, ::dilation] = x
    return y


def _dilate2d_bwd(dy: Array, dilation: int) -> Array:
    return dy[..., ::dilation, ::dilation]


class Dilate2D(Op):
    """2D dilation."""

    def forward(self, x: Array, *, dilation: int) -> Array:
        y = _dilate2d_fwd(self.xp, x, dilation)
        self.stash(dilation)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (dilation,) = self.unstash()
        dx = _dilate2d_bwd(dy, dilation)
        return (dx,)


def _windowed_view_2d(xp: ModuleType, x: Array, window_size: int, stride: int = 1) -> Array:
    # compute strided shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)

    # compute strides
    xstr = x.strides
    out_strides = (*xstr[:-2], xstr[-2] * stride, xstr[-1] * stride, *xstr[-2:])

    return xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)


class Conv2D(Op):
    """2D convolution."""

    def forward(self, x: Array, w: Array, *, stride: int) -> Array:
        kernel_size = w.shape[-1]

        # convolve
        x_windowed = _windowed_view_2d(self.xp, x, kernel_size, stride)
        y = einsum("biyxjk,oijk->boyx", x_windowed, w, use_blas=True)

        self.stash(x, w, x.shape[-1], kernel_size, stride)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, w, input_size, kernel_size, stride = self.unstash()

        # dilate and full pad
        if stride > 1:
            dy = _dilate2d_fwd(self.xp, dy, stride)
        output_size = input_size - kernel_size + 1
        output_shape = (*dy.shape[:-2], output_size, output_size)
        dy = _pad_to_shape(self.xp, dy, output_shape)
        dy = _pad2d_fwd(self.xp, dy, kernel_size - 1)

        # input grad
        w = self.xp.flip(w, axis=(-2, -1))
        dy_windowed = _windowed_view_2d(self.xp, dy, kernel_size)
        dx = einsum("boyxjk,oijk->biyx", dy_windowed, w, use_blas=True)

        # weight grad
        dy_windowed = _windowed_view_2d(self.xp, dy, input_size)
        dw = einsum("bojkyx,biyx->oijk", dy_windowed, x, use_blas=True)
        dw = self.xp.flip(dw, axis=(-2, -1))

        return dx, dw


class ConvTranspose2D(Op):
    """2D transposed convolution."""

    def forward(self, x: Array, w: Array, *, stride: int) -> Array:
        kernel_size = w.shape[-1]

        # dilate and full pad
        if stride > 1:
            x = _dilate2d_fwd(self.xp, x, stride)
        x = _pad2d_fwd(self.xp, x, kernel_size - 1)

        # convolve
        w = self.xp.flip(w, axis=(-2, -1))
        x_pooled = _windowed_view_2d(self.xp, x, kernel_size, 1)
        y = einsum("biyxjk,oijk->boyx", x_pooled, w, use_blas=True)

        self.stash(x, w, x.shape[-1], kernel_size, stride)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, w, input_size, kernel_size, stride = self.unstash()

        # full pad
        dy = _pad2d_fwd(self.xp, dy, kernel_size - 1)

        # input grad
        w = self.xp.flip(w, axis=(-2, -1))
        dy_pooled = _windowed_view_2d(self.xp, dy, kernel_size)
        dx = einsum("boyxjk,oijk->biyx", dy_pooled, w, use_blas=True)
        dx = _pad2d_bwd(dx, kernel_size - 1)
        dx = _dilate2d_bwd(dx, stride)

        # weight grad
        dy_pooled = _windowed_view_2d(self.xp, dy, input_size)
        dw = einsum("bojkyx,biyx->oijk", dy_pooled, x, use_blas=True)

        return dx, dw


def _repeat2d(xp: ModuleType, x: Array, n_repeats: int, target_shape: ShapeLike):
    out_shape = (*x.shape[:-1], n_repeats, x.shape[-1], n_repeats)
    out_strides = (*x.strides[:-1], 0, x.strides[-1], 0)
    y = xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)
    y = y.reshape((*y.shape[:-4], y.shape[-4] * n_repeats, y.shape[-2] * n_repeats))
    return y if y.shape == target_shape else _pad_to_shape(xp, y, target_shape)


class Maxpool2D(Op):
    """2D max pooling."""

    def forward(self, x: Array, *, window_size: int) -> Array:
        y = _windowed_view_2d(self.xp, x, window_size, window_size).max((-2, -1))
        self.stash(x, window_size, y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, window_size, y = self.unstash()
        mask = _repeat2d(self.xp, y, window_size, x.shape) == x
        dy_upsampled = _repeat2d(self.xp, dy, window_size, x.shape)
        dx = dy_upsampled * mask
        return (dx,)


# -------------------------------------------------------------------------------------
# ATTENTION OPERATIONS
# -------------------------------------------------------------------------------------


class ScaledDotProductAttention(Op):
    """Scaled dot-product attention."""

    def forward(
        self,
        q: Array,
        k: Array,
        v: Array,
        attn_mask: Optional[Array],
        *,
        dropout_p: float,
    ) -> Array:
        *_, seq_len, head_size = q.shape

        attn = q @ k.swapaxes(-1, -2) / math.sqrt(head_size)  # q @ k.T / scale
        if attn_mask is not None:
            attn += attn_mask[:seq_len, :seq_len]

        attn_weights = _softmax_fwd(self.xp, attn, dim=-1)

        if dropout_p > 0:
            dropout_mask = _get_dropout_mask(self.xp, attn.shape, dropout_p)
            attn_weights_drop = _dropout_fwd(attn_weights, dropout_mask, dropout_p)
            y = attn_weights_drop @ v
            self.stash(q, k, v, attn_weights, attn_weights_drop, dropout_p, dropout_mask)
        else:
            y = attn_weights @ v
            self.stash(q, k, v, attn_weights, None, 0.0, None)

        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        q, k, v, attn_weights, attn_weights_drop, dropout_p, dropout_mask = self.unstash()
        head_size = q.shape[-1]

        # attention gradients
        dattn_weights = dy @ v.swapaxes(-1, -2)  # dy @ v.T

        if dropout_p > 0:
            dattn_weights = _dropout_bwd(dattn_weights, dropout_mask, dropout_p)

        dattn = _softmax_bwd(attn_weights, dattn_weights, dim=-1)
        dattn /= math.sqrt(head_size)

        # query, key, value gradients
        dq = dattn @ k
        dk = dattn.swapaxes(-1, -2) @ q  # dattn.T @ q

        attn_weights_used = attn_weights_drop if dropout_p > 0 else attn_weights
        dv = attn_weights_used.swapaxes(-1, -2) @ dy  # attn_weights.T @ dy

        return dq, dk, dv, None


# -------------------------------------------------------------------------------------
# NORMALIZATION OPERATIONS
# -------------------------------------------------------------------------------------


class Batchnorm(Op):
    """Batch normalization."""

    def forward(
        self,
        x: Array,
        w: Array,
        b: Array,
        running_mean: Array,
        running_var: Array,
        *,
        momentum: float,
        eps: float,
        training: bool,
    ) -> Array:
        batch_dims = (0, *tuple(d for d in range(x.ndim) if d > 1))
        ext_shape = (1,) * (x.ndim - 2)
        n = x.size / w.size

        if training:
            x_mean = x.mean(batch_dims, keepdims=True)
            x_centered = x - x_mean
            x_var = (x_centered * x_centered).mean(batch_dims, keepdims=True)

            # update rmean and rvar inplace to avoid returning multiple arrays
            running_mean *= 1 - momentum
            running_mean += x_mean.squeeze() * momentum
            running_var *= 1 - momentum
            running_var += n / (n - 1) * x_var.squeeze() * momentum
        else:
            x_mean = running_mean.reshape(*running_mean.shape, *ext_shape)
            x_centered = x - x_mean
            x_var = running_var.reshape(*running_var.shape, *ext_shape)

        x_inv_std = 1 / self.xp.sqrt(x_var + eps)
        x_norm = x_centered * x_inv_std
        w = w.reshape(*w.shape, *ext_shape)
        b = b.reshape(*b.shape, *ext_shape)
        y = x_norm * w + b

        self.stash(w, batch_dims, x_inv_std, x_norm)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        w, batch_dims, x_inv_std, x_norm = self.unstash()

        # bias grad
        db = dy.sum(batch_dims)

        # weight grad
        dw = (dy * x_norm).sum(batch_dims)

        # input grad
        dx_norm = dy * w
        dx = x_inv_std * (
            dx_norm
            - dx_norm.mean(batch_dims, keepdims=True)
            - x_norm * (dx_norm * x_norm).mean(batch_dims, keepdims=True)
        )

        return dx, dw, db, None, None


class Layernorm(Op):
    """Layer normalization."""

    def forward(self, x: Array, w: Array, b: Array, *, eps: float) -> Array:
        feat_dims = tuple(range(x.ndim - w.ndim, x.ndim))

        x_mean = x.mean(feat_dims, keepdims=True)
        x_centered = x - x_mean
        x_var = (x_centered * x_centered).mean(feat_dims, keepdims=True)
        x_inv_std = 1 / self.xp.sqrt(x_var + eps)
        x_norm = x_centered * x_inv_std
        y = x_norm * w + b

        self.stash(w, feat_dims, x_inv_std, x_norm)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        w, feat_dims, x_inv_std, x_norm = self.unstash()
        batch_dims = tuple(range(dy.ndim - w.ndim))

        # bias grad
        db = dy.sum(batch_dims)

        # weight grad
        dw = (dy * x_norm).sum(batch_dims)

        # input grad
        dx_norm = dy * w
        dx = x_inv_std * (
            dx_norm
            - dx_norm.mean(feat_dims, keepdims=True)
            - x_norm * (dx_norm * x_norm).mean(feat_dims, keepdims=True)
        )

        return dx, dw, db


# -------------------------------------------------------------------------------------
# REGULARIZATION OPERATIONS
# -------------------------------------------------------------------------------------


def _get_dropout_mask(xp: ModuleType, shape: ShapeLike, p: float) -> Array:
    return xp.random.rand(*shape) > p


def _dropout_fwd(x: Array, dropout_mask: Array, p: float) -> Array:
    return x * dropout_mask / (1 - p)


def _dropout_bwd(dy: Array, dropout_mask: Array, p: float) -> Array:
    return dy * dropout_mask / (1 - p)


class Dropout(Op):
    """Dropout regularization."""

    def forward(self, x: Array, *, p: float) -> Array:
        dropout_mask = _get_dropout_mask(self.xp, x.shape, p)
        y = _dropout_fwd(x, dropout_mask, p)
        self.stash(p, dropout_mask)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        p, dropout_mask = self.unstash()
        dx = _dropout_bwd(dy, dropout_mask, p)
        return (dx,)


# -------------------------------------------------------------------------------------
# EMBEDDING OPERATIONS
# -------------------------------------------------------------------------------------


class Embedding(Select):
    """Lookup Embedding."""


# -------------------------------------------------------------------------------------
# LOSS FUNCTION OPERATIONS
# -------------------------------------------------------------------------------------


class MSELoss(Op):
    """Mean squared error loss function."""

    def forward(self, x: Array, y: Array, *, reduction: str) -> Array:
        diff = x - y
        loss = diff * diff
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.stash(x.size, diff, reduction)
        return loss

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_size, diff, reduction = self.unstash()
        dx = dy * 2.0 * diff
        dx = dx if reduction != "mean" else dx / float(x_size)
        return dx, None


def _onehot(xp: ModuleType, x: Array, n: int, dtype: type):
    return xp.eye(n, dtype=dtype)[x]


class CrossEntropyLoss(Op):
    """Cross entropy loss function."""

    def forward(self, x: Array, y: Array, *, eta: float, reduction: str) -> Array:
        probs = _softmax_fwd(self.xp, x, dim=-1)
        y_onehot = _onehot(self.xp, y, x.shape[-1], probs.dtype)
        loss = -(self.xp.log(probs + eta) * y_onehot).sum(-1)
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.stash(y, probs, reduction)
        return loss

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y, probs, reduction = self.unstash()
        y = _onehot(self.xp, y, probs.shape[-1], probs.dtype)
        dx = dy * (probs - y)
        dx = dx if reduction != "mean" else dx / float(math.prod(y.shape[:-1]))
        return dx, None


class BCELoss(Op):
    """Binary cross entropy loss function."""

    def forward(self, x: Array, y: Array, *, reduction: str) -> Array:
        max_logits = self.xp.maximum(x, 0.0)
        loss = max_logits - x * y + self.xp.log(1 + self.xp.exp(-self.xp.abs(x)))
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.stash(x, y, reduction)
        return loss

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, y, reduction = self.unstash()
        dx = dy * (_sigmoid_fwd(self.xp, x) - y)
        dx = dx if reduction != "mean" else dx / float(x.size)
        return dx, None
