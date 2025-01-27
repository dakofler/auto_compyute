"""Neural network autograd functions"""

from types import ModuleType

from ..devices import Array
from ..funcs.function import Function


class Softmax(Function):
    def forward(self, x: Array, dim: int) -> Array:
        x = self.m.exp(x - x.max(dim, keepdims=True))
        y = x / x.sum(dim, keepdims=True)
        self.ctx.save_for_backward(dim, y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim, y = self.ctx.get_saved_vals()
        dx = y * (output_grad - (output_grad * y).sum(dim, keepdims=True))
        return (dx,)


def _pad2d_forward(m: ModuleType, x: Array, padding: int) -> Array:
    widths = tuple([(0, 0)] * (x.ndim - 2) + [(padding, padding)] * 2)
    y = m.pad(x, widths)
    return y


class Pad2D(Function):
    def forward(self, x: Array, padding: int) -> Array:
        self.ctx.save_for_backward(padding)
        return _pad2d_forward(self.m, x, padding)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        padding = self.ctx.get_saved_vals()
        dx = output_grad[..., padding:-padding, padding:-padding]
        dx = self.m.to_contiguous(dx)
        return (dx,)


def _dilate2d_forward(m: ModuleType, x: Array, dilation: int) -> Array:
    y_height = dilation * (x.shape[-2] - 1) + 1
    y_width = dilation * (x.shape[-1] - 1) + 1
    y = m.zeros((*x.shape[:-2], y_height, y_width), dtype=x.dtype)
    y[..., ::dilation, ::dilation] = x
    return y


class Dilate2D(Function):
    def forward(self, x: Array, dilation: int) -> Array:
        self.ctx.save_for_backward(dilation)
        return _dilate2d_forward(self.m, x, dilation)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dilation = self.ctx.get_saved_vals()
        dx = output_grad[..., ::dilation, ::dilation]
        return (dx,)


def _pool2d(m: ModuleType, x: Array, window_size: int, stride: int) -> Array:
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)
    x_str = x.strides
    out_strides = (*x_str[:-2], x_str[-2] * stride, x_str[-1] * stride, *x_str[-2:])
    return m.lib.stride_tricks.as_strided(x.data, out_shape, out_strides)


class Conv2D(Function):
    def forward(self, x: Array, w: Array, stride: int) -> Array:
        x_windowed = _pool2d(self.m, x, w.shape[-1], stride)
        y = self.m.einsum("biyxjk,oijk->boyx", x_windowed, w)
        y = self.m.ascontiguousarray(y)
        self.ctx.save_for_backward(x, w, stride)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, w, stride = self.ctx.get_saved_vals()

        # fill elements skipped by strides with zeros
        if stride > 1:
            output_grad = _dilate2d_forward(self.m, output_grad, stride)

        # pad to match unstrided dy
        output_padding = output_grad.shape[-1] - x.shape[-1] + w.shape[-1] - 1
        if output_padding > 0:
            output_grad = _pad2d_forward(self.m, output_grad, output_padding)

        # full pad
        output_grad = _pad2d_forward(self.m, output_grad, w.shape[-1] - 1)

        # input grads
        dy_pooled = _pool2d(self.m, output_grad, w.shape[-1], w.shape[-1])
        w = self.m.flip(w, dim=(-2, -1))
        dx = self.m.einsum("boyxjk,oijk->biyx", dy_pooled, w).to_contiguous()

        # filter grads
        dy_pooled = _pool2d(self.m, output_grad, x.shape[-1], x.shape[-1])
        dw = self.m.einsum("bojkyx,biyx->oijk", dy_pooled, x)
        dw = self.m.flip(dw, dim=(-2, -1)).to_contiguous()

        return dx, dw


class Maxpool2D(Function):
    def forward(self, x: Array, window_size: int) -> Array:
        x_windowed = _pool2d(self.m, x, window_size, window_size)
        y = x_windowed.max((-2, -1))
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        raise NotImplementedError()
