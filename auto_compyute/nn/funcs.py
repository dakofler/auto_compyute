"""Neural network autograd functions"""

import math
from types import ModuleType
from typing import Optional

from ..backends import Array, Shape
from ..funcs.binary_funcs import Maximum
from ..funcs.function import Function
from ..funcs.shape_funcs import Select

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


class GELU(Function):
    def forward(self, x: Array) -> Array:
        tanh_term = self.backend.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x))
        y = 0.5 * x * (1.0 + tanh_term)
        self.cache.save(x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x = self.cache.retrieve()
        # recompute to save memory
        tanh_term = self.backend.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
        dx1 = 1.0 + tanh_term
        dx2 = x * (1.0 - tanh_term * tanh_term) * (0.7978845608 + 0.1070322243 * x * x)
        dx = dy * 0.5 * (dx1 + dx2)
        return (dx,)


class ReLU(Maximum):
    def forward(self, x1: Array) -> Array:
        y = self.backend.maximum(x1, 0.0)
        self.cache.save(True, y == x1)
        return y


def _softmax_forward(backend: ModuleType, x: Array, dim: int) -> Array:
    x = backend.exp(x - x.max(dim, keepdims=True))
    return x / x.sum(dim, keepdims=True)


class Sigmoid(Function):
    def forward(self, x: Array) -> Array:
        y = 1.0 / (1.0 + self.backend.exp(-x))
        self.cache.save(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.cache.retrieve()
        dx = dy * y * (1.0 - y)
        return (dx,)


class Softmax(Function):
    def forward(self, x: Array, dim: int) -> Array:
        y = _softmax_forward(self.backend, x, dim)
        self.cache.save(dim, y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, y = self.cache.retrieve()
        dx = y * (dy - (dy * y).sum(dim, keepdims=True))
        return (dx,)


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


class Linear(Function):
    def forward(self, x: Array, w: Array, b: Optional[Array]) -> Array:
        self.cache.save(x, w, b is None)
        y = x @ w.swapaxes(-1, -2)
        if b is not None:
            y += b
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, w, b_is_none = self.cache.retrieve()
        dx = dy @ w
        dw = (dy.swapaxes(-1, -2) @ x).sum(tuple(range(dy.ndim - 2)))
        if b_is_none:
            return dx, dw
        db = dy.sum(tuple(range(dy.ndim - 1)))
        return dx, dw, db


# -------------------------------------------------------------------------------------
# CONVOLUTION FUNCTIONS
# -------------------------------------------------------------------------------------


def _pad2d_forward(backend: ModuleType, x: Array, padding: int) -> Array:
    widths = tuple([(0, 0)] * (x.ndim - 2) + [(padding, padding)] * 2)
    y = backend.pad(x, widths)
    return y


class Pad2D(Function):
    def forward(self, x: Array, padding: int) -> Array:
        self.cache.save(padding)
        return _pad2d_forward(self.backend, x, padding)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        padding = self.cache.retrieve()
        dx = dy[..., padding:-padding, padding:-padding]
        dx = self.backend.ascontiguousarray(dx)
        return (dx,)


def _dilate2d_forward(backend: ModuleType, x: Array, dilation: int) -> Array:
    y_height = dilation * (x.shape[-2] - 1) + 1
    y_width = dilation * (x.shape[-1] - 1) + 1
    y = backend.zeros((*x.shape[:-2], y_height, y_width), dtype=x.dtype)
    y[..., ::dilation, ::dilation] = x
    return y


class Dilate2D(Function):
    def forward(self, x: Array, dilation: int) -> Array:
        self.cache.save(dilation)
        return _dilate2d_forward(self.backend, x, dilation)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dilation = self.cache.retrieve()
        dx = dy[..., ::dilation, ::dilation]
        return (dx,)


def _pool2d(backend: ModuleType, x: Array, window_size: int, stride: int = 1) -> Array:
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)
    x_str = x.strides
    out_strides = (*x_str[:-2], x_str[-2] * stride, x_str[-1] * stride, *x_str[-2:])
    return backend.lib.stride_tricks.as_strided(x, out_shape, out_strides)


def _pad_to_shape(backend: ModuleType, x: Array, shape: Shape) -> Array:
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    return backend.pad(x, padding)


class Conv2D(Function):
    def forward(self, x: Array, w: Array, stride: int) -> Array:
        self.cache.save(x, w, stride)
        x_pooled = _pool2d(self.backend, x, w.shape[-1], stride)
        y = self.backend.einsum("biyxjk,oijk->boyx", x_pooled, w)
        return self.backend.ascontiguousarray(y)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, w, stride = self.cache.retrieve()
        *_, input_size = x.shape
        *_, kernel_size = w.shape

        # fill elements skipped by strides with zeros
        if stride > 1:
            dy = _dilate2d_forward(self.backend, dy, stride)

        # pad to match unstrided dy
        output_size = input_size - kernel_size + 1
        output_shape = (*dy.shape[:-2], output_size, output_size)
        dy = _pad_to_shape(self.backend, dy, output_shape)

        # full pad
        dy = _pad2d_forward(self.backend, dy, kernel_size - 1)

        # input grads
        dy_pooled = _pool2d(self.backend, dy, kernel_size)
        w = self.backend.flip(w, (-2, -1))
        dx = self.backend.einsum("boyxjk,oijk->biyx", dy_pooled, w)
        dx = self.backend.ascontiguousarray(dx)

        # filter grads
        dy_pooled = _pool2d(self.backend, dy, input_size)
        dw = self.backend.einsum("bojkyx,biyx->oijk", dy_pooled, x)
        dw = self.backend.flip(dw, (-2, -1))
        dw = self.backend.ascontiguousarray(dw)

        return dx, dw


def _repeat2d(backend: ModuleType, x: Array, n_repeats: int, target_shape: Shape):
    repeat_shape = (*x.shape[:-1], n_repeats, x.shape[-1], n_repeats)
    repeat_strides = (*x.strides[:-1], 0, x.strides[-1], 0)
    y = backend.lib.stride_tricks.as_strided(x, repeat_shape, repeat_strides)
    y = y.reshape((*y.shape[:-4], y.shape[-4] * n_repeats, y.shape[-2] * n_repeats))
    if y.shape != target_shape:
        y = _pad_to_shape(backend, y, target_shape)
    return y


class Maxpool2D(Function):
    def forward(self, x: Array, window_size: int) -> Array:
        y = _pool2d(self.backend, x, window_size, window_size).max((-2, -1))
        self.cache.save(x, window_size, y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, window_size, y = self.cache.retrieve()
        mask = _repeat2d(self.backend, y, window_size, x.shape) == x
        dx = _repeat2d(self.backend, dy, window_size, x.shape) * mask
        return (dx,)


# -------------------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Layernorm(Function):
    def forward(self, x: Array, w: Array, b: Array, eps: float) -> Array:
        f_dims = tuple(range(x.ndim - w.ndim, x.ndim))
        n = w.size

        mean = x.sum(f_dims, keepdims=True) / n
        xshift = x - mean
        var = (xshift * xshift).sum(f_dims, keepdims=True) / n
        rstd = (var + eps) ** -0.5
        x_norm = xshift * rstd
        y = x_norm * w + b

        self.cache.save(w, f_dims, rstd, x_norm)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        w, f_dims, rstd, x_norm = self.cache.retrieve()
        b_dims = tuple(range(dy.ndim - w.ndim))

        db = dy.sum(b_dims)
        dw = (dy * x_norm).sum(b_dims)
        dx_norm = dy * w
        dx = rstd * (
            dx_norm
            - dx_norm.mean(f_dims, keepdims=True)
            - x_norm * (dx_norm * x_norm).mean(f_dims, keepdims=True)
        )
        return dx, dw, db


# -------------------------------------------------------------------------------------
# REGULARIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Dropout(Function):
    def forward(self, x: Array, p: float) -> Array:
        p = 1.0 - p
        dropout_mask = self.backend.random.random(x.shape) < p
        self.cache.save(p, dropout_mask)
        return x * dropout_mask / p

    def backward(self, dy: Array) -> tuple[Array, ...]:
        p, dropout_mask = self.cache.retrieve()
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
    def forward(self, x: Array, y: Array) -> Array:
        diff = x - y
        self.cache.save(x.size, diff)
        return (diff * diff).mean()

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_size, diff = self.cache.retrieve()
        dx = dy * 2.0 * diff / float(x_size)
        return (dx,)


def _onehot(backend: ModuleType, x: Array, n: int, dtype: type):
    return backend.eye(n, dtype=dtype)[x]


class CrossEntropyLoss(Function):
    def forward(self, x: Array, y: Array, eta: float) -> Array:
        probs = _softmax_forward(self.backend, x, dim=-1)
        self.cache.save(y, probs)
        y = _onehot(self.backend, y, x.shape[-1], probs.dtype)
        return -(self.backend.log(probs + eta) * y).sum(-1).mean()

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y, probs = self.cache.retrieve()
        y = _onehot(self.backend, y, probs.shape[-1], probs.dtype)
        dx = dy * (probs - y) / float(math.prod(y.shape[:-1]))
        return (dx,)
