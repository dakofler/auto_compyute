"""Reduce autograd functions"""

from typing import Optional

from ..backends import Array
from .function import Function


class Sum(Function):
    def forward(
        self, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        self.ctx.save_for_backward(x.shape)
        return x.sum(dim, keepdims=keepdims)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        shape = self.ctx.get_saved_vals()
        dx = self.m.broadcast_to(output_grad, shape)
        return (dx,)


class Mean(Function):
    def forward(
        self, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        y = x.mean(dim, keepdims=keepdims)
        self.ctx.save_for_backward(x.shape, x.size / y.size)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x_shape, size = self.ctx.get_saved_vals()
        dx = self.m.broadcast_to(output_grad / size, x_shape)
        return (dx,)


class Var(Function):
    def forward(
        self, x: Array, dim: Optional[int | tuple[int, ...]], ddof: int, keepdims: bool
    ) -> Array:
        y = x.var(dim, ddof=ddof, keepdims=keepdims)
        self.ctx.save_for_backward(x, dim, x.size / y.size - ddof)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, dim, size = self.ctx.get_saved_vals()
        mean = x.mean(dim)
        dx = output_grad * (2 / size) * (x - mean)
        return (dx,)


class Std(Function):
    def forward(
        self, x: Array, dim: Optional[int | tuple[int, ...]], ddof: int, keepdims: bool
    ) -> Array:
        y = x.std(dim, ddof=ddof, keepdims=keepdims)
        self.ctx.save_for_backward(x, dim, ddof, y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, dim, ddof, y = self.ctx.get_saved_vals()
        mean = x.mean(dim)
        size = x.size / y.size - ddof
        dx = output_grad * (x - mean) / (size * y)
        return (dx,)


class Max(Function):
    def forward(
        self, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        y = x.max(dim, keepdims=keepdims)
        self.ctx.save_for_backward(dim, x == y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim, mask = self.ctx.get_saved_vals()
        dx = mask * output_grad / mask.sum(dim, keepdims=True)
        return (dx,)


class Min(Function):
    def forward(
        self, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        raise NotImplementedError()

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        raise NotImplementedError()
