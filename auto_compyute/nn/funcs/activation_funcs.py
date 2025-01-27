"""Activation autograd functions"""

from ...devices import Array
from ...funcs.function import Function


class Softmax(Function):
    def forward(self, x: Array, dim: int) -> Array:
        x = self.m.exp(x - x.max(dim, keepdims=True))
        y = x / x.sum(dim, keepdims=True)
        self.ctx.save_for_backward(dim, y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim, y = self.ctx.get_saved_vals()
        dx = y * (output_grad - (output_grad * y).sum(dim, keepdims=True))
        return (dx,)
