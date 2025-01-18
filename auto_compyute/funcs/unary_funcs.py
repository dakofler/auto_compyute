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
