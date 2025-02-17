"""Neural network differentiable operations."""

import math
from types import ModuleType
from typing import Optional

from opt_einsum import contract as einsum  # type: ignore

from ..backends import ArrayLike, ShapeLike
from ..ops.op import Op
from ..ops.movement_ops import Select

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTION OPERATIONS
# -------------------------------------------------------------------------------------


class GELU(Op):
    """Gaussian error linear unit activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = 0.5 * x * (1 + self.xp.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
        if x_req_grad:
            self.save_to_cache(x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (x,) = self.retrieve_from_cache()
        t = self.xp.tanh(x * 0.79788 * (1 + 0.04472 * x * x))
        dx = dy * 0.5 * ((1 + t) + x * (1 - t * t) * (0.79788 + 0.10703 * x * x))
        return (dx,)


class ReLU(Op):
    """Rectified linear unit activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = self.xp.maximum(x, 0.0)
        if x_req_grad:
            self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (mask,) = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)


class LeakyReLU(Op):
    """Leaky rectified linear unit activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, alpha: float) -> ArrayLike:
        y = self.xp.maximum(x, x * alpha)
        if x_req_grad:
            self.save_to_cache(alpha, y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        alpha, mask = self.retrieve_from_cache()
        dx = dy * (mask + (~mask).astype(dy.dtype) * alpha)
        return (dx,)


def _sigmoid_forward(xp: ModuleType, x: ArrayLike) -> ArrayLike:
    return 1 / (1 + xp.exp(-x))


class Sigmoid(Op):
    """Sigmoid activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = _sigmoid_forward(self.xp, x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * y * (1 - y)
        return (dx,)


def _softmax_forward(xp: ModuleType, x: ArrayLike, dim: int) -> ArrayLike:
    x = xp.exp(x - x.max(dim, keepdims=True))
    return x / x.sum(dim, keepdims=True)


def _softmax_backward(y: ArrayLike, dy: ArrayLike, dim: int) -> ArrayLike:
    return y * (dy - (dy * y).sum(dim, keepdims=True))


class Softmax(Op):
    """Softmax activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, dim: int) -> ArrayLike:
        y = _softmax_forward(self.xp, x, dim)
        if x_req_grad:
            self.save_to_cache(dim, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, y = self.retrieve_from_cache()
        dx = _softmax_backward(y, dy, dim)
        return (dx,)


# -------------------------------------------------------------------------------------
# LINEAR OPERATIONS
# -------------------------------------------------------------------------------------


class Linear(Op):
    """Linear projection."""

    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        w: ArrayLike,
        w_req_grad: bool,
        b: Optional[ArrayLike],
        b_req_grad: bool,
    ) -> ArrayLike:
        y = x @ w.swapaxes(-1, -2)
        y = y if b is None else y + b
        self.save_to_cache(
            (x if w_req_grad else None),
            (w if x_req_grad else None),
            b_req_grad,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, b_requires_grad = self.retrieve_from_cache()
        dx = None if w is None else (dy @ w)
        dw = None if x is None else (dy.swapaxes(-1, -2) @ x)
        db = None if not b_requires_grad else dy
        return dx, dw, db


# -------------------------------------------------------------------------------------
# CONVOLUTION OPERATIONS
# -------------------------------------------------------------------------------------


def _pad1d_forward(
    xp: ModuleType, x: ArrayLike, left_pad: int, right_pad: Optional[int] = None
) -> ArrayLike:
    right_pad = right_pad if right_pad is not None else left_pad
    paddings = tuple([(0, 0)] * (x.ndim - 1) + [(left_pad, right_pad)])
    return xp.pad(x, paddings)


def _pad1d_backward(
    dy: ArrayLike, left_pad: int, right_pad: Optional[int] = None
) -> ArrayLike:
    right_pad = right_pad if right_pad is not None else left_pad
    if right_pad <= 0:
        return dy[..., left_pad:]
    return dy[..., left_pad:-right_pad]


class Pad1D(Op):
    """1D padding."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, padding: int) -> ArrayLike:
        y = _pad1d_forward(self.xp, x, padding, padding)
        if x_req_grad:
            self.save_to_cache(padding)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (padding,) = self.retrieve_from_cache()
        dx = _pad1d_backward(dy, padding, padding)
        return (dx,)


def _dilate1d_forward(xp: ModuleType, x: ArrayLike, dilation: int) -> ArrayLike:
    y_size = dilation * (x.shape[-1] - 1) + 1
    y_shape = (*x.shape[:-1], y_size)
    y = xp.zeros(y_shape, dtype=x.dtype)
    y[..., ::dilation] = x
    return y


def _dilate1d_backward(dy: ArrayLike, dilation: int) -> ArrayLike:
    return dy[..., ::dilation]


class Dilate1D(Op):
    """1D dilation."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, dilation: int) -> ArrayLike:
        y = _dilate1d_forward(self.xp, x, dilation)
        if x_req_grad:
            self.save_to_cache(dilation)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (dilation,) = self.retrieve_from_cache()
        dx = _dilate1d_backward(dy, dilation)
        return (dx,)


def _pad_to_shape(xp: ModuleType, x: ArrayLike, shape: ShapeLike) -> ArrayLike:
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    return xp.pad(x, padding)


def _windowed_view_1d(
    xp: ModuleType, x: ArrayLike, window_size: int, stride: int = 1
) -> ArrayLike:
    # compute strided shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-1], out, window_size)

    # compute strides
    xstr = x.strides
    out_strides = (*xstr[:-1], xstr[-1] * stride, xstr[-1])

    return xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)


class Conv1D(Op):
    """1D convolution."""

    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        w: ArrayLike,
        w_req_grad: bool,
        *,
        stride: int,
    ) -> ArrayLike:
        kernel_size = w.shape[-1]

        # convolve
        x_windowed = _windowed_view_1d(self.xp, x, kernel_size, stride)
        y = einsum("bitk,oik->bot", x_windowed, w, use_blas=True)

        self.save_to_cache(
            (x if w_req_grad else None),
            (w if x_req_grad else None),
            x.shape[-1],  # input size
            kernel_size,
            stride,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, input_size, kernel_size, stride = self.retrieve_from_cache()

        # dilate and full pad
        if stride > 1:
            dy = _dilate1d_forward(self.xp, dy, stride)
        output_size = input_size - kernel_size + 1
        output_shape = (*dy.shape[:-1], output_size)
        dy = _pad_to_shape(self.xp, dy, output_shape)
        dy = _pad1d_forward(self.xp, dy, kernel_size - 1)

        # input grads
        if w is not None:
            w = self.xp.flip(w, axis=-1)
            dy_windowed = _windowed_view_1d(self.xp, dy, kernel_size)
            dx = einsum("botk,oik->bit", dy_windowed, w, use_blas=True)
        else:
            dx = None

        # weight grads
        if x is not None:
            dy_windowed = _windowed_view_1d(self.xp, dy, input_size)
            dw = einsum("bokt,bit->oik", dy_windowed, x, use_blas=True)
            dw = self.xp.flip(dw, axis=-1)
        else:
            dw = None

        return dx, dw


def _pad2d_forward(
    xp: ModuleType, x: ArrayLike, left_pad: int, right_pad: Optional[int] = None
) -> ArrayLike:
    right_pad = right_pad if right_pad is not None else left_pad
    paddings = tuple([(0, 0)] * (x.ndim - 2) + [(left_pad, right_pad)] * 2)
    return xp.pad(x, paddings)


def _pad2d_backward(
    dy: ArrayLike, left_pad: int, right_pad: Optional[int] = None
) -> ArrayLike:
    right_pad = right_pad if right_pad is not None else left_pad
    if right_pad <= 0:
        return dy[..., left_pad:, left_pad:]
    return dy[..., left_pad:-right_pad, left_pad:-right_pad]


class Pad2D(Op):
    """2D padding."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, padding: int) -> ArrayLike:
        y = _pad2d_forward(self.xp, x, padding, padding)
        if x_req_grad:
            self.save_to_cache(padding)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (padding,) = self.retrieve_from_cache()
        dx = _pad2d_backward(dy, padding, padding)
        return (dx,)


class OutPad2D(Op):
    """2D output padding."""

    def forward(
        self, x: ArrayLike, x_req_grad: bool, *, padding: int, output_padding: int
    ) -> ArrayLike:
        y = _pad2d_backward(x, padding, padding - output_padding)
        if x_req_grad:
            self.save_to_cache(padding, output_padding)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        padding, output_padding = self.retrieve_from_cache()
        dx = _pad2d_forward(self.xp, dy, padding, padding - output_padding)
        return (dx,)


def _dilate2d_forward(xp: ModuleType, x: ArrayLike, dilation: int) -> ArrayLike:
    y_size = dilation * (x.shape[-1] - 1) + 1
    y_shape = (*x.shape[:-2], y_size, y_size)
    y = xp.zeros(y_shape, dtype=x.dtype)
    y[..., ::dilation, ::dilation] = x
    return y


def _dilate2d_backward(dy: ArrayLike, dilation: int) -> ArrayLike:
    return dy[..., ::dilation, ::dilation]


class Dilate2D(Op):
    """2D dilation."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, dilation: int) -> ArrayLike:
        y = _dilate2d_forward(self.xp, x, dilation)
        if x_req_grad:
            self.save_to_cache(dilation)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (dilation,) = self.retrieve_from_cache()
        dx = _dilate2d_backward(dy, dilation)
        return (dx,)


def _windowed_view_2d(
    xp: ModuleType, x: ArrayLike, window_size: int, stride: int = 1
) -> ArrayLike:
    # compute strided shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)

    # compute strides
    xstr = x.strides
    out_strides = (*xstr[:-2], xstr[-2] * stride, xstr[-1] * stride, *xstr[-2:])

    return xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)


class Conv2D(Op):
    """2D convolution."""

    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        w: ArrayLike,
        w_req_grad: bool,
        *,
        stride: int,
    ) -> ArrayLike:
        kernel_size = w.shape[-1]

        # convolve
        x_windowed = _windowed_view_2d(self.xp, x, kernel_size, stride)
        y = einsum("biyxjk,oijk->boyx", x_windowed, w, use_blas=True)

        self.save_to_cache(
            (x if w_req_grad else None),
            (w if x_req_grad else None),
            x.shape[-1],  # input size
            kernel_size,
            stride,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, input_size, kernel_size, stride = self.retrieve_from_cache()

        # dilate and full pad
        if stride > 1:
            dy = _dilate2d_forward(self.xp, dy, stride)
        output_size = input_size - kernel_size + 1
        output_shape = (*dy.shape[:-2], output_size, output_size)
        dy = _pad_to_shape(self.xp, dy, output_shape)
        dy = _pad2d_forward(self.xp, dy, kernel_size - 1)

        # input grads
        if w is not None:
            w = self.xp.flip(w, axis=(-2, -1))
            dy_windowed = _windowed_view_2d(self.xp, dy, kernel_size)
            dx = einsum("boyxjk,oijk->biyx", dy_windowed, w, use_blas=True)
        else:
            dx = None

        # weight grads
        if x is not None:
            dy_windowed = _windowed_view_2d(self.xp, dy, input_size)
            dw = einsum("bojkyx,biyx->oijk", dy_windowed, x, use_blas=True)
            dw = self.xp.flip(dw, axis=(-2, -1))
        else:
            dw = None

        return dx, dw


class ConvTranspose2D(Op):
    """2D transposed convolution."""

    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        w: ArrayLike,
        w_req_grad: bool,
        *,
        stride: int,
    ) -> ArrayLike:
        kernel_size = w.shape[-1]

        # dilate and full pad
        if stride > 1:
            x = _dilate2d_forward(self.xp, x, stride)
        x = _pad2d_forward(self.xp, x, kernel_size - 1)

        # convolve
        w = self.xp.flip(w, axis=(-2, -1))
        x_pooled = _windowed_view_2d(self.xp, x, kernel_size, 1)
        y = einsum("biyxjk,oijk->boyx", x_pooled, w, use_blas=True)

        self.save_to_cache(
            (x if w_req_grad else None),
            (w if x_req_grad else None),
            x.shape[-1],  # input size
            kernel_size,
            stride,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, input_size, kernel_size, stride = self.retrieve_from_cache()

        # full pad
        dy = _pad2d_forward(self.xp, dy, kernel_size - 1)

        # input grads
        if w is not None:
            w = self.xp.flip(w, axis=(-2, -1))
            dy_pooled = _windowed_view_2d(self.xp, dy, kernel_size)
            dx = einsum("boyxjk,oijk->biyx", dy_pooled, w, use_blas=True)
            dx = _pad2d_backward(dx, kernel_size - 1)
            dx = _dilate2d_backward(dx, stride)
        else:
            dx = None

        # weight grads
        if x is not None:
            dy_pooled = _windowed_view_2d(self.xp, dy, input_size)
            dw = einsum("bojkyx,biyx->oijk", dy_pooled, x, use_blas=True)
        else:
            dw = None

        return dx, dw


def _repeat2d(xp: ModuleType, x: ArrayLike, n_repeats: int, target_shape: ShapeLike):
    out_shape = (*x.shape[:-1], n_repeats, x.shape[-1], n_repeats)
    out_strides = (*x.strides[:-1], 0, x.strides[-1], 0)
    y = xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)
    y = y.reshape((*y.shape[:-4], y.shape[-4] * n_repeats, y.shape[-2] * n_repeats))
    return y if y.shape == target_shape else _pad_to_shape(xp, y, target_shape)


class Maxpool2D(Op):
    """2D max pooling."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, window_size: int) -> ArrayLike:
        y = _windowed_view_2d(self.xp, x, window_size, window_size).max((-2, -1))
        if x_req_grad:
            self.save_to_cache(x, window_size, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, window_size, y = self.retrieve_from_cache()
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
        q: ArrayLike,
        q_req_grad: bool,
        k: ArrayLike,
        k_req_grad: bool,
        v: ArrayLike,
        v_req_grad: bool,
        attn_mask: Optional[ArrayLike],
        _1: bool,  # dummy placehodler for mask_req_grad
        *,
        p: float,
    ) -> ArrayLike:
        *_, seq_len, head_size = q.shape

        attn = q @ k.swapaxes(-1, -2) / math.sqrt(head_size)
        if attn_mask is not None:
            attn += attn_mask[:seq_len, :seq_len]
        attnw = _softmax_forward(self.xp, x=attn, dim=-1)
        drop_mask = None if p == 0 else _get_dropout_mask(self.xp, attn.shape, p)
        drop_attnw = attnw if p == 0 else _dropout_forward(attnw, drop_mask, p)
        y = drop_attnw @ v

        self.save_to_cache(
            (q if k_req_grad else None),
            (k if q_req_grad else None),
            (v if q_req_grad or k_req_grad else None),
            drop_attnw,
            p,
            drop_mask,
            v_req_grad,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        q, k, v, drop_attnw, p, drop_mask, v_req_grad = self.retrieve_from_cache()
        head_size = q.shape[-1]

        # attention gradients
        dattn_weights = dy @ v.swapaxes(-1, -2)
        if p > 0:
            dattn_weights = _dropout_backward(dattn_weights, drop_mask, p)
        dattn_weights = _softmax_backward(drop_attnw, dattn_weights, -1)
        dattn_weights /= math.sqrt(head_size)

        # query, key, value gradients
        dq = None if k is None else (dattn_weights @ k)
        dk = None if q is None else (dattn_weights.swapaxes(-1, -2) @ q)
        dv = None if not v_req_grad else (drop_attnw.swapaxes(-1, -2) @ dy)

        return dq, dk, dv


# -------------------------------------------------------------------------------------
# NORMALIZATION OPERATIONS
# -------------------------------------------------------------------------------------


class Batchnorm(Op):
    """Batch normalization."""

    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        w: ArrayLike,
        w_req_grad: bool,
        b: ArrayLike,
        b_req_grad: bool,
        rmean: ArrayLike,
        _1: bool,  # dummy placeholder for rmean_req_grad
        rvar: ArrayLike,
        _2: bool,  # dummy placeholder for rvar_req_grad
        *,
        momentum: float,
        eps: float,
        training: bool,
    ) -> ArrayLike:
        b_dims = (0,) + tuple(d for d in range(x.ndim) if d > 1)
        ext_shape = (1,) * (x.ndim - 2)
        n = x.size / w.size

        if training:
            mean = x.mean(b_dims, keepdims=True)
            xshift = x - mean
            var = (xshift * xshift).mean(b_dims, keepdims=True)

            # update rmean and rvar inplace to avoid returning multiple arrays
            rmean *= 1 - momentum
            rmean += mean.squeeze() * momentum
            rvar *= 1 - momentum
            rvar += n / (n - 1) * var.squeeze() * momentum
        else:
            mean = rmean.reshape(*rmean.shape, *ext_shape)
            xshift = x - mean
            var = rvar.reshape(*rvar.shape, *ext_shape)

        rstd = 1 / self.xp.sqrt(var + eps)
        xnorm = xshift * rstd
        w = w.reshape(*w.shape, *ext_shape)
        b = b.reshape(*b.shape, *ext_shape)
        y = xnorm * w + b

        self.save_to_cache(
            (w if x_req_grad else None),
            b_dims,
            (rstd if x_req_grad else None),
            (xnorm if x_req_grad or w_req_grad else None),
            b_req_grad,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        w, b_dims, rstd, xnorm, b_req_grad = self.retrieve_from_cache()

        # bias grads
        db = None if not b_req_grad else dy.sum(b_dims)

        # weight grads
        dw = None if xnorm is None else (dy * xnorm).sum(b_dims)

        # input grads
        if rstd is None:
            dx = None
        else:
            dxnorm = dy * w
            dx = rstd * (
                dxnorm
                - dxnorm.mean(b_dims, keepdims=True)
                - xnorm * (dxnorm * xnorm).mean(b_dims, keepdims=True)
            )

        return dx, dw, db


class Layernorm(Op):
    """Layer normalization."""

    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        w: ArrayLike,
        w_req_grad: bool,
        b: ArrayLike,
        b_req_grad: bool,
        *,
        eps: float,
    ) -> ArrayLike:
        f_dims = tuple(range(x.ndim - w.ndim, x.ndim))

        mean = x.mean(f_dims, keepdims=True)
        xshift = x - mean
        var = (xshift * xshift).mean(f_dims, keepdims=True)
        rstd = 1 / self.xp.sqrt(var + eps)
        xnorm = xshift * rstd
        y = xnorm * w + b

        self.save_to_cache(
            (w if x_req_grad else None),
            f_dims,
            (rstd if x_req_grad else None),
            (xnorm if x_req_grad or w_req_grad else None),
            b_req_grad,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        w, f_dims, rstd, xnorm, b_req_grad = self.retrieve_from_cache()
        b_dims = tuple(range(dy.ndim - w.ndim))

        # bias grads
        db = None if not b_req_grad else dy.sum(b_dims)

        # weight grads
        dw = None if xnorm is None else (dy * xnorm).sum(b_dims)

        # input grads
        if rstd is None:
            dx = None
        else:
            dxnorm = dy * w
            dx = rstd * (
                dxnorm
                - dxnorm.mean(f_dims, keepdims=True)
                - xnorm * (dxnorm * xnorm).mean(f_dims, keepdims=True)
            )

        return dx, dw, db


# -------------------------------------------------------------------------------------
# REGULARIZATION OPERATIONS
# -------------------------------------------------------------------------------------


def _get_dropout_mask(xp: ModuleType, shape: ShapeLike, p: float) -> ArrayLike:
    return xp.random.rand(*shape) > p


def _dropout_forward(x: ArrayLike, dropout_mask: ArrayLike, p: float) -> ArrayLike:
    return x * dropout_mask / (1 - p)


def _dropout_backward(dy: ArrayLike, dropout_mask: ArrayLike, p: float) -> ArrayLike:
    return dy * dropout_mask / (1 - p)


class Dropout(Op):
    """Dropout regularization."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, p: float) -> ArrayLike:
        dropout_mask = _get_dropout_mask(self.xp, x.shape, p)
        y = _dropout_forward(x, dropout_mask, p)
        if x_req_grad:
            self.save_to_cache(p, dropout_mask)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        p, dropout_mask = self.retrieve_from_cache()
        dx = _dropout_backward(dy, dropout_mask, p)
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

    def forward(
        self, x: ArrayLike, x_req_grad: bool, y: ArrayLike, _: bool, *, reduction: str
    ) -> ArrayLike:
        diff = x - y
        loss = diff * diff
        loss = loss.mean() if reduction == "mean" else loss.sum()
        if x_req_grad:
            self.save_to_cache(x.size, diff, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_size, diff, reduction = self.retrieve_from_cache()
        dx = dy * 2.0 * diff
        if reduction == "mean":
            dx /= float(x_size)
        return (dx,)


def _onehot(xp: ModuleType, x: ArrayLike, n: int, dtype: type):
    return xp.eye(n, dtype=dtype)[x]


class CrossEntropyLoss(Op):
    """Cross entropy loss function."""

    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        y: ArrayLike,
        _: bool,
        *,
        eta: float,
        reduction: str,
    ) -> ArrayLike:
        probs = _softmax_forward(self.xp, x, dim=-1)
        y_onehot = _onehot(self.xp, y, x.shape[-1], probs.dtype)
        loss = -(self.xp.log(probs + eta) * y_onehot).sum(-1)
        loss = loss.mean() if reduction == "mean" else loss.sum()
        if x_req_grad:
            self.save_to_cache(y, probs, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        y, probs, reduction = self.retrieve_from_cache()
        y = _onehot(self.xp, y, probs.shape[-1], probs.dtype)
        dx = dy * (probs - y)
        if reduction == "mean":
            dx /= float(math.prod(y.shape[:-1]))
        return (dx,)


class BCELoss(Op):
    """Binary cross entropy loss function."""

    def forward(
        self, x: ArrayLike, x_req_grad: bool, y: ArrayLike, _: bool, *, reduction: str
    ) -> ArrayLike:
        max_logits = self.xp.maximum(x, 0.0)
        loss = max_logits - x * y + self.xp.log(1 + self.xp.exp(-self.xp.abs(x)))
        loss = loss.mean() if reduction == "mean" else loss.sum()
        if x_req_grad:
            self.save_to_cache(x, y, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, y, reduction = self.retrieve_from_cache()
        dx = dy * (_sigmoid_forward(self.xp, x) - y)
        if reduction == "mean":
            dx /= float(x.size)
        return (dx,)
