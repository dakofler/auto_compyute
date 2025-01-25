"""Multiary autograd functions"""

from ..backends import Array, Scalar, get_array_backend
from .function import Function


class Pow(Function):
    def forward(self, x: Array, exp: Scalar) -> Array:
        self.ctx.x, self.ctx.exp = x, exp
        return x**exp

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, exp = self.ctx.x, self.ctx.exp
        dx = output_grad * exp * x ** (exp - 1)
        return (dx,)


class Tanh(Function):
    def forward(self, x: Array) -> Array:
        b = get_array_backend(x).m
        y = b.tanh(x)
        self.ctx.y = y
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dx = output_grad * (1 - self.ctx.y**2)
        return (dx,)
