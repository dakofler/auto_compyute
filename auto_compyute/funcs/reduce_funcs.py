"""Reduce autograd functions"""

from typing import Optional

from ..backends import Array, get_array_backend
from .function import Function


class Sum(Function):
    def forward(
        self, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        self.ctx.shape = x.shape
        return x.sum(dim, keepdims=keepdims)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        b = get_array_backend(output_grad).m
        dx = b.broadcast_to(output_grad, self.ctx.shape)
        return (dx,)


class Mean(Function):
    def forward(
        self, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        self.ctx.shape = x.shape
        return x.mean(dim, keepdims=keepdims)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        b = get_array_backend(output_grad).m
        dx = b.broadcast_to(output_grad / b.prod(self.ctx.shape), self.ctx.shape)
        return (dx,)


class Var(Function):
    def forward(self, x: Array) -> Array:
        raise NotImplementedError()

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        raise NotImplementedError()


class Std(Function):
    def forward(self, x: Array) -> Array:
        raise NotImplementedError()

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        raise NotImplementedError()
