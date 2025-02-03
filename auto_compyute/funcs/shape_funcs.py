"""Shape autograd functions"""

from itertools import accumulate
from typing import Any

from ..backends import Array, Shape
from .function import Function


class Concat(Function):
    def forward(self, *arrays: Array, dim: int) -> Array:
        self.save_to_cache(dim, [a.shape[dim] for a in arrays])
        return self.xp.concatenate(arrays, dim)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, split_sizes = self.retrieve_from_cache()
        split_indices = list(accumulate(s for s in split_sizes))
        return tuple(self.xp.split(dy, split_indices, dim))


class Select(Function):
    def forward(self, x: Array, key: Any) -> Array:
        self.save_to_cache(x.shape, key)
        return x[key]

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape, key = self.retrieve_from_cache()
        dx = self.xp.zeros(x_shape, dtype=dy.dtype)
        self.xp.add.at(dx, key, dy)
        return (dx,)


class Split(Select): ...


class Stack(Function):
    def forward(self, *arrays: Array, dim: int) -> Array:
        self.save_to_cache(dim)
        return self.xp.stack(arrays, dim)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim = self.retrieve_from_cache()
        return tuple(self.xp.moveaxis(dy, dim, 0))


class Transpose(Function):
    def forward(self, x: Array, dim1, dim2) -> Array:
        self.save_to_cache(dim1, dim2)
        return x.swapaxes(dim1, dim2)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim1, dim2 = self.retrieve_from_cache()
        dx = dy.swapaxes(dim1, dim2)
        return (dx,)


class View(Function):
    def forward(self, x: Array, shape: Shape) -> Array:
        self.save_to_cache(x.shape)
        return x.reshape(shape)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape = self.retrieve_from_cache()
        dx = dy.reshape(x_shape)
        return (dx,)


class Squeeze(View): ...
