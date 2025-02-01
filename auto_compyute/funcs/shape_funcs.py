"""Shape autograd functions"""

from itertools import accumulate
from typing import Any

from ..backends import Array, Shape
from .function import Function


class Concat(Function):
    def forward(self, *arrays: Array, dim: int) -> Array:
        self.ctx.save(dim, [a.shape[dim] for a in arrays])
        return self.m.concatenate(arrays, dim)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim, split_sizes = self.ctx.retrieve()
        split_indices = list(accumulate(s for s in split_sizes))
        return tuple(self.m.split(output_grad, split_indices, dim))


class Select(Function):
    def forward(self, x: Array, key: Any) -> Array:
        self.ctx.save(x.shape, key)
        return x[key]

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        shape, key = self.ctx.retrieve()
        dx = self.m.zeros(shape, dtype=output_grad.dtype)
        self.m.add.at(dx, key, output_grad)
        return (dx,)


class Split(Select): ...


class Stack(Function):
    def forward(self, *arrays: Array, dim: int) -> Array:
        self.ctx.save(dim)
        return self.m.stack(arrays, dim)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim = self.ctx.retrieve()
        return tuple(self.m.moveaxis(output_grad, dim, 0))


class Transpose(Function):
    def forward(self, x: Array, dim1, dim2) -> Array:
        self.ctx.save(dim1, dim2)
        return x.swapaxes(dim1, dim2)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim1, dim2 = self.ctx.retrieve()
        dx = output_grad.swapaxes(dim1, dim2)
        return (dx,)


class View(Function):
    def forward(self, x: Array, shape: Shape) -> Array:
        self.ctx.save(x.shape)
        return x.reshape(shape)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        shape = self.ctx.retrieve()
        dx = output_grad.reshape(shape)
        return (dx,)
