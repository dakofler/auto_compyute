"""Shape autograd functions"""

from itertools import accumulate
from typing import Any

from ..backends import Array, ShapeLike
from .function import Function


class Concat(Function):
    def forward(self, *arrays_and_req_grads: Array | bool, dim: int) -> Array:
        arrays = arrays_and_req_grads[::2]
        req_grads = arrays_and_req_grads[1::2]
        y = self.xp.concatenate(arrays, dim)
        self.save_to_cache(dim, req_grads, [a.shape[dim] for a in arrays])
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, req_grads, split_sizes = self.retrieve_from_cache()
        split_indices = list(accumulate(s for s in split_sizes))
        dxs = self.xp.split(dy, split_indices, dim)
        return tuple(dx if req_grad else None for dx, req_grad in zip(dxs, req_grads))


class Expand(Function):
    def forward(self, x: Array, _: bool, *, shape: ShapeLike) -> Array:
        y = self.xp.broadcast_to(x, shape)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        return (dy,)


class Select(Function):
    def forward(self, x: Array, x_req_grad: bool, *, key: Any) -> Array:
        y = x[key]
        if x_req_grad:
            self.save_to_cache(x.shape, key)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape, key = self.retrieve_from_cache()
        dx = self.xp.zeros(x_shape, dtype=dy.dtype)
        self.xp.add.at(dx, key, dy)
        return (dx,)


class Split(Select): ...


class Stack(Function):
    def forward(self, *arrays_and_req_grads: Array | bool, dim: int) -> Array:
        arrays = arrays_and_req_grads[::2]
        req_grads = arrays_and_req_grads[1::2]
        y = self.xp.stack(arrays, dim)
        self.save_to_cache(dim, req_grads)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, req_grads = self.retrieve_from_cache()
        dxs = tuple(self.xp.moveaxis(dy, dim, 0))
        return tuple(dx if req_grad else None for dx, req_grad in zip(dxs, req_grads))


class Transpose(Function):
    def forward(self, x: Array, x_req_grad: bool, *, dim1, dim2) -> Array:
        y = x.swapaxes(dim1, dim2)
        if x_req_grad:
            self.save_to_cache(dim1, dim2)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim1, dim2 = self.retrieve_from_cache()
        dx = dy.swapaxes(dim1, dim2)
        return (dx,)


class View(Function):
    def forward(self, x: Array, x_req_grad: bool, *, shape: ShapeLike) -> Array:
        y = self.xp.reshape(x, shape)
        if x_req_grad:
            self.save_to_cache(x.shape)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape = self.retrieve_from_cache()
        dx = dy.reshape(x_shape)
        return (dx,)


class Squeeze(View): ...


class Where(Function):
    def forward(
        self,
        condition: Array,
        _: bool,
        x1: Array,
        x1_req_grad: bool,
        x2: Array,
        x2_req_grad: bool,
    ) -> Array:
        y = self.xp.where(condition, x1, x2)
        self.save_to_cache(
            x1_req_grad,
            x2_req_grad,
            ((y == x1) if x1_req_grad or x2_req_grad else None),
        )
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x1_req_grad, x2_req_grad, mask = self.retrieve_from_cache()
        dx1 = None if not x1_req_grad else (dy * mask)
        dx2 = None if not x2_req_grad else (dy * self.xp.invert(mask))
        return None, dx1, dx2
