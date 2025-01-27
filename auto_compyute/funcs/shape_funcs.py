"""Shape autograd functions"""

from typing import Any

from ..devices import Array, Shape
from .function import Function


class Select(Function):
    def forward(self, x: Array, key: Any) -> Array:
        self.ctx.save_for_backward(x.shape, key)
        return x[key]

    def backward(self, grad: Array) -> tuple[Array, ...]:
        shape, key = self.ctx.get_saved_vals()
        dx = self.m.zeros(shape, dtype=grad.dtype)
        self.m.add.at(dx, key, grad)
        return (dx,)


class Transpose(Function):
    def forward(self, x: Array, dim1, dim2) -> Array:
        self.ctx.save_for_backward(dim1, dim2)
        return x.swapaxes(dim1, dim2)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim1, dim2 = self.ctx.get_saved_vals()
        dx = output_grad.swapaxes(dim1, dim2)
        return (dx,)


class View(Function):
    def forward(self, x: Array, shape: Shape) -> Array:
        self.ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        shape = self.ctx.get_saved_vals()
        dx = output_grad.reshape(shape)
        return (dx,)
