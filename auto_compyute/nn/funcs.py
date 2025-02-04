"""Neural network autograd functions"""

import math
from types import ModuleType
from typing import Optional

import opt_einsum as oe  # type: ignore

from ..backends import Array, Shape
from ..funcs.function import Function
from ..funcs.shape_funcs import Select

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


class GELU(Function):

    def forward(self, x: Array, x_req_grad: bool) -> Array:
        tanh_term = self.xp.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x))
        y = 0.5 * x * (1.0 + tanh_term)
        if x_req_grad:
            self.save_to_cache(x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x = self.retrieve_from_cache()
        tanh_term = self.xp.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
        dx1 = 1.0 + tanh_term
        dx2 = x * (1.0 - tanh_term * tanh_term) * (0.7978845608 + 0.1070322243 * x * x)
        dx = dy * 0.5 * (dx1 + dx2)
        return (dx,)


class ReLU(Function):
    def forward(self, x: Array, x_req_grad: bool) -> Array:
        y = self.xp.maximum(x, 0.0)
        if x_req_grad:
            self.save_to_cache(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)


class LeakyReLU(Function):
    def forward(self, x: Array, x_req_grad: bool, *, alpha: float) -> Array:
        y = self.xp.maximum(x, x * alpha)
        if x_req_grad:
            self.save_to_cache(alpha, y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        alpha, mask = self.retrieve_from_cache()
        dx = dy * (mask + (~mask).astype(dy.dtype) * alpha)
        return (dx,)


def _sigmoid_forward(xp: ModuleType, x: Array) -> Array:
    return 1.0 / (1.0 + xp.exp(-x))


class Sigmoid(Function):
    def forward(self, x: Array, x_req_grad: bool) -> Array:
        y = _sigmoid_forward(self.xp, x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.retrieve_from_cache()
        dx = dy * y * (1.0 - y)
        return (dx,)


def _softmax_forward(xp: ModuleType, x: Array, dim: int) -> Array:
    x = xp.exp(x - x.max(dim, keepdims=True))
    x = x / x.sum(dim, keepdims=True)
    return x


class Softmax(Function):
    def forward(self, x: Array, x_req_grad: bool, *, dim: int) -> Array:
        y = _softmax_forward(self.xp, x, dim)
        if x_req_grad:
            self.save_to_cache(dim, y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, y = self.retrieve_from_cache()
        dx = y * (dy - (dy * y).sum(dim, keepdims=True))
        return (dx,)

        # -------------------------------------------------------------------------------------
        # LINEAR FUNCTIONS
        # -------------------------------------------------------------------------------------


class Linear(Function):
    def forward(
        self,
        x: Array,
        x_req_grad: bool,
        w: Array,
        w_req_grad: bool,
        b: Optional[Array],
        b_req_grad: bool,
    ) -> Array:
        y = x @ w.swapaxes(-1, -2)
        y = y if b is None else y + b
        self.save_to_cache(
            (x if w_req_grad else None),
            (w if x_req_grad else None),
            b_req_grad,
        )
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, w, b_requires_grad = self.retrieve_from_cache()
        dx = None if w is None else (dy @ w)
        dw = None if x is None else (dy.swapaxes(-1, -2) @ x)
        db = None if not b_requires_grad else dy
        return dx, dw, db


# -------------------------------------------------------------------------------------
# CONVOLUTION FUNCTIONS
# -------------------------------------------------------------------------------------


def _pad2d_forward(xp: ModuleType, x: Array, padding: int) -> Array:
    widths = tuple([(0, 0)] * (x.ndim - 2) + [(padding, padding)] * 2)
    y = xp.pad(x, widths)
    return y


class Pad2D(Function):
    def forward(self, x: Array, x_req_grad: bool, *, padding: int) -> Array:
        y = _pad2d_forward(self.xp, x, padding)
        if x_req_grad:
            self.save_to_cache(padding)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        padding = self.retrieve_from_cache()
        dx = dy[..., padding:-padding, padding:-padding]
        dx = self.xp.ascontiguousarray(dx)
        return (dx,)


def _dilate2d_forward(xp: ModuleType, x: Array, dilation: int) -> Array:
    y_height = dilation * (x.shape[-2] - 1) + 1
    y_width = dilation * (x.shape[-1] - 1) + 1
    y = xp.zeros((*x.shape[:-2], y_height, y_width), dtype=x.dtype)
    y[..., ::dilation, ::dilation] = x
    return y


class Dilate2D(Function):
    def forward(self, x: Array, x_req_grad: bool, *, dilation: int) -> Array:
        y = _dilate2d_forward(self.xp, x, dilation)
        if x_req_grad:
            self.save_to_cache(dilation)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dilation = self.retrieve_from_cache()
        dx = dy[..., ::dilation, ::dilation]
        return (dx,)


def _pool2d(xp: ModuleType, x: Array, window_size: int, stride: int = 1) -> Array:
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)
    xstr = x.strides
    out_strides = (*xstr[:-2], xstr[-2] * stride, xstr[-1] * stride, *xstr[-2:])
    y = xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)
    return y


def _pad_to_shape(xp: ModuleType, x: Array, shape: Shape) -> Array:
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    y = xp.pad(x, padding)
    return y


class Conv2D(Function):
    def forward(
        self, x: Array, x_req_grad: bool, w: Array, w_req_grad: bool, *, stride: int
    ) -> Array:
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

    def backward(self, dy: Array) -> tuple[Array, ...]:
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


def _repeat2d(xp: ModuleType, x: Array, n_repeats: int, target_shape: Shape):
    repeat_shape = (*x.shape[:-1], n_repeats, x.shape[-1], n_repeats)
    repeat_strides = (*x.strides[:-1], 0, x.strides[-1], 0)
    y = xp.lib.stride_tricks.as_strided(x, repeat_shape, repeat_strides)
    y = y.reshape((*y.shape[:-4], y.shape[-4] * n_repeats, y.shape[-2] * n_repeats))
    y = y if y.shape == target_shape else _pad_to_shape(xp, y, target_shape)
    return y


class Maxpool2D(Function):
    def forward(self, x: Array, x_req_grad: bool, *, window_size: int) -> Array:
        y = _pool2d(self.xp, x, window_size, window_size).max((-2, -1))
        if x_req_grad:
            self.save_to_cache(x, window_size, y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, window_size, y = self.retrieve_from_cache()
        mask = _repeat2d(self.xp, y, window_size, x.shape) == x
        dx = _repeat2d(self.xp, dy, window_size, x.shape) * mask
        return (dx,)


# -------------------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Batchnorm(Function):
    def forward(
        self,
        x: Array,
        x_req_grad: bool,
        w: Array,
        w_req_grad: bool,
        b: Array,
        b_req_grad: bool,
        rmean: Array,
        _1: bool,  # dummy placeholders for rmean_req_grad
        rvar: Array,
        _2: bool,  # dummy placeholders for rvar_req_grad
        *,
        momentum: float,
        eps: float,
        training: bool,
    ) -> Array:
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

    def backward(self, dy: Array) -> tuple[Array, ...]:
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
    def forward(
        self,
        x: Array,
        x_req_grad: bool,
        w: Array,
        w_req_grad: bool,
        b: Array,
        b_req_grad: bool,
        *,
        eps: float,
    ) -> Array:
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

    def backward(self, dy: Array) -> tuple[Array, ...]:
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
    def forward(self, x: Array, x_req_grad: bool, *, p: float) -> Array:
        p = 1.0 - p
        dropout_mask = self.xp.random.random(x.shape) < p
        y = x * dropout_mask / p
        if x_req_grad:
            self.save_to_cache(p, dropout_mask)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        p, dropout_mask = self.retrieve_from_cache()
        dx = dy * dropout_mask / p
        return (dx,)


# -------------------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
# -------------------------------------------------------------------------------------


class Embedding(Select): ...


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


class MSELoss(Function):
    def forward(
        self, x: Array, x_req_grad: bool, y: Array, _: bool, *, reduction: str
    ) -> Array:
        diff = x - y
        loss = diff * diff
        loss = loss.mean() if reduction == "mean" else loss.sum()
        if x_req_grad:
            self.save_to_cache(x.size, diff, reduction)
        return loss

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_size, diff, reduction = self.retrieve_from_cache()
        dx = dy * 2.0 * diff
        if reduction == "mean":
            dx /= float(x_size)
        return (dx,)


def _onehot(xp: ModuleType, x: Array, n: int, dtype: type):
    return xp.eye(n, dtype=dtype)[x]


class CrossEntropyLoss(Function):
    def forward(
        self,
        x: Array,
        x_req_grad: bool,
        y: Array,
        _: bool,
        *,
        eta: float,
        reduction: str,
    ) -> Array:
        probs = _softmax_forward(self.xp, x, dim=-1)
        y_onehot = _onehot(self.xp, y, x.shape[-1], probs.dtype)
        loss = -(self.xp.log(probs + eta) * y_onehot).sum(-1)
        loss = loss.mean() if reduction == "mean" else loss.sum()
        if x_req_grad:
            self.save_to_cache(y, probs, reduction)
        return loss

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y, probs, reduction = self.retrieve_from_cache()
        y = _onehot(self.xp, y, probs.shape[-1], probs.dtype)
        dx = dy * (probs - y)
        if reduction == "mean":
            dx /= float(math.prod(y.shape[:-1]))
        return (dx,)


class BCELoss(Function):
    def forward(
        self, x: Array, x_req_grad: bool, y: Array, _: bool, *, reduction: str
    ) -> Array:
        max_logits = self.xp.maximum(x, 0.0)
        loss = max_logits - x * y + self.xp.log(1.0 + self.xp.exp(-self.xp.abs(x)))
        loss = loss.mean() if reduction == "mean" else loss.sum()
        if x_req_grad:
            self.save_to_cache(x, y, reduction)
        return loss

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, y, reduction = self.retrieve_from_cache()
        dx = dy * (_sigmoid_forward(self.xp, x) - y)
        if reduction == "mean":
            dx /= float(x.size)
        return (dx,)
