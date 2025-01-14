"""Multiary autograd functions"""

from ..backends import Array, get_array_backend
from .function import Context, Function


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, x: Array) -> Array:
        b = get_array_backend(x).m
        y = b.tanh(x)
        ctx.save(y)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        y = ctx.get()
        dx = output_grad * (1 - y**2)
        return (dx,)


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, x: Array, dim1, dim2) -> Array:
        y = x.swapaxes(dim1, dim2)
        ctx.save(dim1, dim2)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        dim1, dim2 = ctx.get()
        dx = output_grad.swapaxes(dim1, dim2)
        return (dx,)
