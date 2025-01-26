"""Multiary autograd functions"""

from ..backends import Array, Scalar
from .function import Function


class Exp(Function):
    def forward(self, x: Array) -> Array:
        y = self.m.exp(x)
        self.ctx.save_for_backward(y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        y = self.ctx.get_saved_vals()
        dx = output_grad * y
        return (dx,)


class Pow(Function):
    def forward(self, x: Array, exp: Scalar) -> Array:
        self.ctx.save_for_backward(x, exp)
        return x**exp

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, exp = self.ctx.get_saved_vals()
        dx = output_grad * exp * x ** (exp - 1)
        return (dx,)


class Tanh(Function):
    def forward(self, x: Array) -> Array:
        y = self.m.tanh(x)
        self.ctx.save_for_backward(y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        y = self.ctx.get_saved_vals()
        dx = output_grad * (1 - y**2)
        return (dx,)
