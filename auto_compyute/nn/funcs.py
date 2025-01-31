"""Neural network autograd functions"""

from types import ModuleType

from ..backends import Array, Shape
from ..funcs.function import Function

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Softmax(Function):
    def forward(self, x: Array, dim: int) -> Array:
        x = self.m.exp(x - x.max(dim, keepdims=True))
        y = x / x.sum(dim, keepdims=True)
        self.ctx.save(dim, y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim, y = self.ctx.retrieve()
        dx = y * (output_grad - (output_grad * y).sum(dim, keepdims=True))
        return (dx,)


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


class Linear(Function):
    def forward(self, x: Array, w: Array, b: Array) -> Array:
        y = x @ w.swapaxes(-2, -1) + b
        self.ctx.save(x, w)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, w = self.ctx.retrieve()
        dx = output_grad @ w
        dw = (output_grad.swapaxes(-2, -1) @ x).sum(tuple(range(output_grad.ndim - 2)))
        db = output_grad.sum(tuple(range(output_grad.ndim - 1)))
        return dx, dw, db


# -------------------------------------------------------------------------------------
# CONVOLUTION FUNCTIONS
# -------------------------------------------------------------------------------------


def _pad2d_forward(m: ModuleType, x: Array, padding: int) -> Array:
    widths = tuple([(0, 0)] * (x.ndim - 2) + [(padding, padding)] * 2)
    y = m.pad(x, widths)
    return y


class Pad2D(Function):
    def forward(self, x: Array, padding: int) -> Array:
        self.ctx.save(padding)
        return _pad2d_forward(self.m, x, padding)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        padding = self.ctx.retrieve()
        dx = output_grad[..., padding:-padding, padding:-padding]
        dx = self.m.ascontiguousarray(dx)
        return (dx,)


def _dilate2d_forward(m: ModuleType, x: Array, dilation: int) -> Array:
    y_height = dilation * (x.shape[-2] - 1) + 1
    y_width = dilation * (x.shape[-1] - 1) + 1
    y = m.zeros((*x.shape[:-2], y_height, y_width), dtype=x.dtype)
    y[..., ::dilation, ::dilation] = x
    return y


class Dilate2D(Function):
    def forward(self, x: Array, dilation: int) -> Array:
        self.ctx.save(dilation)
        return _dilate2d_forward(self.m, x, dilation)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dilation = self.ctx.retrieve()
        dx = output_grad[..., ::dilation, ::dilation]
        return (dx,)


def _pool2d(m: ModuleType, x: Array, window_size: int, stride: int = 1) -> Array:
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)
    x_str = x.strides
    out_strides = (*x_str[:-2], x_str[-2] * stride, x_str[-1] * stride, *x_str[-2:])
    return m.lib.stride_tricks.as_strided(x, out_shape, out_strides)


def _pad_to_shape(m: ModuleType, x: Array, shape: Shape) -> Array:
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    return m.pad(x, padding)


class Conv2D(Function):
    def forward(self, x: Array, w: Array, stride: int) -> Array:
        self.ctx.save(x, w, stride)
        x_pooled = _pool2d(self.m, x, w.shape[-1], stride)
        y = self.m.einsum("biyxjk,oijk->boyx", x_pooled, w, order="C")
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, w, stride = self.ctx.retrieve()
        *_, input_size = x.shape
        *_, kernel_size = w.shape

        # fill elements skipped by strides with zeros
        if stride > 1:
            output_grad = _dilate2d_forward(self.m, output_grad, stride)

        # pad to match unstrided dy
        output_size = input_size - kernel_size + 1
        output_shape = (*output_grad.shape[:-2], output_size, output_size)
        output_grad = _pad_to_shape(self.m, output_grad, output_shape)

        # full pad
        output_grad = _pad2d_forward(self.m, output_grad, kernel_size - 1)

        # input grads
        output_grad_pooled = _pool2d(self.m, output_grad, kernel_size)
        w = self.m.flip(w, (-2, -1))
        dx = self.m.einsum("boyxjk,oijk->biyx", output_grad_pooled, w, order="C")

        # filter grads
        output_grad_pooled = _pool2d(self.m, output_grad, input_size)
        dw = self.m.einsum("bojkyx,biyx->oijk", output_grad_pooled, x, order="C")
        dw = self.m.flip(dw, (-2, -1))

        return dx, dw


def _repeat2d(m: ModuleType, x: Array, n_repeats: int, target_shape: Shape):
    repeat_shape = (*x.shape[:-1], n_repeats, x.shape[-1], n_repeats)
    repeat_strides = (*x.strides[:-1], 0, x.strides[-1], 0)
    y = m.lib.stride_tricks.as_strided(x.data, repeat_shape, repeat_strides)
    y = y.reshape((*y.shape[:-4], y.shape[-4] * n_repeats, y.shape[-2] * n_repeats))
    if y.shape != target_shape:
        y = _pad_to_shape(m, y, target_shape)
    return y


class Maxpool2D(Function):
    def forward(self, x: Array, window_size: int) -> Array:
        y = _pool2d(self.m, x, window_size, window_size).max((-2, -1))
        self.ctx.save(x, window_size, y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, window_size, y = self.ctx.retrieve()
        mask = _repeat2d(self.m, y, window_size, x.shape) == x
        mask = mask.astype(output_grad.dtype)
        dx = _repeat2d(self.m, output_grad, window_size, x.shape) * mask
        return (dx,)


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


class MSELoss(Function):
    def forward(self, x: Array, y: Array) -> Array:
        diff = x - y
        loss = (diff * diff).mean()
        self.ctx.save(x.size, diff)
        return loss

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x_size, diff = self.ctx.retrieve()
        return output_grad * 2.0 * diff / float(x_size)
