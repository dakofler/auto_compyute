"""Shape autograd functions"""

from itertools import accumulate
from typing import Any

from ..backends import Array, Shape
from .function import Function


class Concat(Function):
    def forward(self, *arrays: Array, dim: int) -> Array:
        self.cache.save(dim, [a.shape[dim] for a in arrays])
        return self.backend.concatenate(arrays, dim)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, split_sizes = self.cache.retrieve()
        split_indices = list(accumulate(s for s in split_sizes))
        return tuple(self.backend.split(dy, split_indices, dim))


class Select(Function):
    def forward(self, x: Array, key: Any) -> Array:
        self.cache.save(x.shape, key)
        return x[key]

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape, key = self.cache.retrieve()
        dx = self.backend.zeros(x_shape, dtype=dy.dtype)
        self.backend.add.at(dx, key, dy)
        return (dx,)


class Split(Select): ...


class Stack(Function):
    def forward(self, *arrays: Array, dim: int) -> Array:
        self.cache.save(dim)
        return self.backend.stack(arrays, dim)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim = self.cache.retrieve()
        return tuple(self.backend.moveaxis(dy, dim, 0))


class Transpose(Function):
    def forward(self, x: Array, dim1, dim2) -> Array:
        self.cache.save(dim1, dim2)
        return x.swapaxes(dim1, dim2)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim1, dim2 = self.cache.retrieve()
        dx = dy.swapaxes(dim1, dim2)
        return (dx,)


class View(Function):
    def forward(self, x: Array, shape: Shape) -> Array:
        self.cache.save(x.shape)
        return x.reshape(shape)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape = self.cache.retrieve()
        dx = dy.reshape(x_shape)
        return (dx,)


class Squeeze(View): ...
