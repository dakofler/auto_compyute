"""Reduce autograd functions"""

from typing import Optional

from ..backends import Array, get_array_backend
from .function import Context, Function


class Sum(Function):
    @staticmethod
    def forward(
        ctx: Context, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        y = x.sum(dim, keepdims=keepdims)
        ctx.save(x.shape)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x_shape = ctx.get()
        b = get_array_backend(output_grad).m
        dx = b.broadcast_to(output_grad, x_shape)
        return (dx,)


class Mean(Function):
    @staticmethod
    def forward(
        ctx: Context, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        y = x.mean(dim, keepdims=keepdims)
        ctx.save(x.shape)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x_shape = ctx.get()
        b = get_array_backend(output_grad).m
        dx = b.broadcast_to(output_grad / b.prod(x_shape), x_shape)
        return (dx,)


class Var(Function):
    @staticmethod
    def forward(ctx: Context, x: Array) -> Array:
        b = get_array_backend(x).m
        raise NotImplementedError()
        ctx.save(y)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        y = ctx.get()
        raise NotImplementedError()
        return (dx,)


class Std(Function):
    @staticmethod
    def forward(ctx: Context, x: Array) -> Array:
        b = get_array_backend(x).m
        raise NotImplementedError()
        ctx.save(y)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        y = ctx.get()
        raise NotImplementedError()
        return (dx,)
