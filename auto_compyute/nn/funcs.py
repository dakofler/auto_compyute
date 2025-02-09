"""Neural network autograd functions."""

import math
from types import ModuleType
from typing import Optional

import opt_einsum as oe  # type: ignore

from ..backends import ArrayLike, ShapeLike
from ..funcs.function import Function
from ..funcs.shape_funcs import Select

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


class GELU(Function):
    """Gaussian error linear unit activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        tanh_term = self.xp.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x))
        y = 0.5 * x * (1.0 + tanh_term)
        if x_req_grad:
            self.save_to_cache(x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x = self.retrieve_from_cache()
        tanh_term = self.xp.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
        dx1 = 1.0 + tanh_term
        dx2 = x * (1.0 - tanh_term * tanh_term) * (0.7978845608 + 0.1070322243 * x * x)
        dx = dy * 0.5 * (dx1 + dx2)
        return (dx,)


class ReLU(Function):
    """Rectified linear unit activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = self.xp.maximum(x, 0.0)
        if x_req_grad:
            self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        mask = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)


class LeakyReLU(Function):
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
    return 1.0 / (1.0 + xp.exp(-x))


class Sigmoid(Function):
    """Sigmoid activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = _sigmoid_forward(self.xp, x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        y = self.retrieve_from_cache()
        dx = dy * y * (1.0 - y)
        return (dx,)


def _softmax_forward(xp: ModuleType, x: ArrayLike, dim: int) -> ArrayLike:
    x = xp.exp(x - x.max(dim, keepdims=True))
    x = x / x.sum(dim, keepdims=True)
    return x


class Softmax(Function):
    """Softmax activation function."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, dim: int) -> ArrayLike:
        y = _softmax_forward(self.xp, x, dim)
        if x_req_grad:
            self.save_to_cache(dim, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, y = self.retrieve_from_cache()
        dx = y * (dy - (dy * y).sum(dim, keepdims=True))
        return (dx,)


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


class Linear(Function):
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
# CONVOLUTION FUNCTIONS
# -------------------------------------------------------------------------------------


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


class Pad2D(Function):
    """2D padding."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, padding: int) -> ArrayLike:
        widths = tuple([(0, 0)] * (x.ndim - 2) + [(padding, padding)] * 2)
        y = self.xp.pad(x, widths)
        if x_req_grad:
            self.save_to_cache(padding)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        padding = self.retrieve_from_cache()
        dx = dy[..., padding:-padding, padding:-padding]
        return (dx,)


class OutPad2D(Function):
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


class Dilate2D(Function):
    """2D dilation."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, dilation: int) -> ArrayLike:
        y = _dilate2d_forward(self.xp, x, dilation)
        if x_req_grad:
            self.save_to_cache(dilation)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dilation = self.retrieve_from_cache()
        dx = _dilate2d_backward(dy, dilation)
        return (dx,)


def _pool2d(
    xp: ModuleType, x: ArrayLike, window_size: int, stride: int = 1
) -> ArrayLike:
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)
    xstr = x.strides
    out_strides = (*xstr[:-2], xstr[-2] * stride, xstr[-1] * stride, *xstr[-2:])
    y = xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)
    return y


def _pad_to_shape(xp: ModuleType, x: ArrayLike, shape: ShapeLike) -> ArrayLike:
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    y = xp.pad(x, padding)
    return y


class Conv2D(Function):
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
        x_pooled = _pool2d(self.xp, x, w.shape[-1], stride)
        y = oe.contract("biyxjk,oijk->boyx", x_pooled, w, use_blas=True)
        self.save_to_cache(
            (x if w_req_grad else None),
            (w if x_req_grad else None),
            x.shape[-1],
            w.shape[-1],
            stride,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, input_size, kernel_size, stride = self.retrieve_from_cache()

        # fill elements skipped by strides with zeros
        if stride > 1:
            dy = _dilate2d_forward(self.xp, dy, stride)

        # pad to match unstrided dy
        output_size = input_size - kernel_size + 1
        output_shape = (*dy.shape[:-2], output_size, output_size)
        dy = _pad_to_shape(self.xp, dy, output_shape)

        # full pad
        dy = _pad2d_forward(self.xp, dy, kernel_size - 1)

        # input grads
        if w is not None:
            dy_pooled = _pool2d(self.xp, dy, kernel_size)
            w = self.xp.flip(w, (-2, -1))
            dx = oe.contract("boyxjk,oijk->biyx", dy_pooled, w, use_blas=True)
        else:
            dx = None

        # weight grads
        if x is not None:
            dy_pooled = _pool2d(self.xp, dy, input_size)
            dw = oe.contract("bojkyx,biyx->oijk", dy_pooled, x, use_blas=True)
            dw = self.xp.flip(dw, (-2, -1))
        else:
            dw = None

        return dx, dw


class ConvTranspose2D(Function):
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
        w = self.xp.flip(w, (-2, -1))

        # upsample input by dilating
        x = _dilate2d_forward(self.xp, x, stride)

        # full pad input
        x = _pad2d_forward(self.xp, x, w.shape[-1] - 1)

        # convolve
        x_pooled = _pool2d(self.xp, x, w.shape[-1], 1)
        y = oe.contract("biyxjk,oijk->boyx", x_pooled, w, use_blas=True)

        self.save_to_cache(
            (x if w_req_grad else None),
            (w if x_req_grad else None),
            x.shape[-1],
            w.shape[-1],
            stride,
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, input_size, kernel_size, stride = self.retrieve_from_cache()

        # full pad
        dy = _pad2d_forward(self.xp, dy, kernel_size - 1)

        # input grads
        if w is not None:
            dy_pooled = _pool2d(self.xp, dy, kernel_size)
            w = self.xp.flip(w, (-2, -1))
            dx = oe.contract("boyxjk,oijk->biyx", dy_pooled, w, use_blas=True)
            dx = _pad2d_backward(dx, kernel_size - 1)
            dx = _dilate2d_backward(dx, stride)
        else:
            dx = None

        # weight grads
        if x is not None:
            dy_pooled = _pool2d(self.xp, dy, input_size)
            dw = oe.contract("bojkyx,biyx->oijk", dy_pooled, x, use_blas=True)
        else:
            dw = None

        return dx, dw


def _repeat2d(xp: ModuleType, x: ArrayLike, n_repeats: int, target_shape: ShapeLike):
    repeat_shape = (*x.shape[:-1], n_repeats, x.shape[-1], n_repeats)
    repeat_strides = (*x.strides[:-1], 0, x.strides[-1], 0)
    y = xp.lib.stride_tricks.as_strided(x, repeat_shape, repeat_strides)
    y = y.reshape((*y.shape[:-4], y.shape[-4] * n_repeats, y.shape[-2] * n_repeats))
    y = y if y.shape == target_shape else _pad_to_shape(xp, y, target_shape)
    return y


class Maxpool2D(Function):
    """2D max pooling."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, window_size: int) -> ArrayLike:
        y = _pool2d(self.xp, x, window_size, window_size).max((-2, -1))
        if x_req_grad:
            self.save_to_cache(x, window_size, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, window_size, y = self.retrieve_from_cache()
        mask = _repeat2d(self.xp, y, window_size, x.shape) == x
        dx = _repeat2d(self.xp, dy, window_size, x.shape) * mask
        return (dx,)


# -------------------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Batchnorm(Function):
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
        _1: bool,  # dummy placeholders for rmean_req_grad
        rvar: ArrayLike,
        _2: bool,  # dummy placeholders for rvar_req_grad
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
            rmean *= 1.0 - momentum
            rmean += mean.squeeze() * momentum
            rvar *= 1.0 - momentum
            rvar += n / (n - 1) * var.squeeze() * momentum
        else:
            mean = rmean.reshape(*rmean.shape, *ext_shape)
            xshift = x - mean
            var = rvar.reshape(*rvar.shape, *ext_shape)

        rstd = (var + eps) ** -0.5
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
        if rstd is not None:
            dxnorm = dy * w
            dx = rstd * (
                dxnorm
                - dxnorm.mean(b_dims, keepdims=True)
                - xnorm * (dxnorm * xnorm).mean(b_dims, keepdims=True)
            )
        else:
            dx = None

        return dx, dw, db


class Layernorm(Function):
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
        rstd = (var + eps) ** -0.5
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
        if rstd is not None:
            dxnorm = dy * w
            dx = rstd * (
                dxnorm
                - dxnorm.mean(f_dims, keepdims=True)
                - xnorm * (dxnorm * xnorm).mean(f_dims, keepdims=True)
            )
        else:
            dx = None

        return dx, dw, db


# -------------------------------------------------------------------------------------
# REGULARIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Dropout(Function):
    """Dropout regularization."""

    def forward(self, x: ArrayLike, x_req_grad: bool, *, p: float) -> ArrayLike:
        p = 1.0 - p
        dropout_mask = self.xp.random.random(x.shape) < p
        y = x * dropout_mask / p
        if x_req_grad:
            self.save_to_cache(p, dropout_mask)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        p, dropout_mask = self.retrieve_from_cache()
        dx = dy * dropout_mask / p
        return (dx,)


# -------------------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
# -------------------------------------------------------------------------------------


class Embedding(Select):
    """Lookup Embedding."""


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


class MSELoss(Function):
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


class CrossEntropyLoss(Function):
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


class BCELoss(Function):
    """Binary cross entropy loss function."""

    def forward(
        self, x: ArrayLike, x_req_grad: bool, y: ArrayLike, _: bool, *, reduction: str
    ) -> ArrayLike:
        max_logits = self.xp.maximum(x, 0.0)
        loss = max_logits - x * y + self.xp.log(1.0 + self.xp.exp(-self.xp.abs(x)))
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
