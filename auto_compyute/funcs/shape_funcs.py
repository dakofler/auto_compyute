"""Shape autograd functions"""

from typing import Any

from ..backends import Array, get_array_backend
from .function import Context, Function


class Select(Function):
    @staticmethod
    def forward(ctx: Context, x: Array, key: Any) -> Array:
        y = x[key]
        ctx.save(x.shape, key)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x_shape, key = ctx.get()
        b = get_array_backend(output_grad).m
        dx = b.zeros(x_shape, dtype=output_grad.dtype)
        b.add.at(dx, key, output_grad)
        return (dx,)
