"""Shape autograd functions"""

from typing import Any

from ..backends import Array, get_array_backend
from .function import Function


class Select(Function):
    def forward(self, x: Array, key: Any) -> Array:
        self.ctx.shape, self.ctx.key = x.shape, key
        return x[key]

    def backward(self, grad: Array) -> tuple[Array, ...]:
        b = get_array_backend(grad).m
        dx = b.zeros(self.ctx.shape, dtype=grad.dtype)
        b.add.at(dx, self.ctx.key, grad)
        return (dx,)


class Transpose(Function):
    def forward(self, x: Array, dim1, dim2) -> Array:
        self.ctx.dim1, self.ctx.dim2 = dim1, dim2
        return x.swapaxes(dim1, dim2)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dx = output_grad.swapaxes(self.ctx.dim1, self.ctx.dim2)
        return (dx,)
