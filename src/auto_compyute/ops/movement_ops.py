"""Movement operations."""

from itertools import accumulate
from typing import Any

from auto_compyute.backends import Array, ShapeLike
from auto_compyute.ops.op import Op


class Concat(Op):
    """Concatinates arrays."""

    def forward(self, *arrays: Array, dim: int) -> Array:
        y = self.xp.concatenate(arrays, dim)
        self.stash(dim, [a.shape[dim] for a in arrays])
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, split_sizes = self.unstash()
        split_indices = list(accumulate(s for s in split_sizes))
        dxs = self.xp.split(dy, split_indices, dim)
        return tuple(dxs)


class Expand(Op):
    """Broadcasts array elements."""

    def forward(self, x: Array, *, shape: ShapeLike) -> Array:
        y = self.xp.broadcast_to(x, shape)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        return (dy,)


class Select(Op):
    """Selects array elements."""

    def forward(self, x: Array, *, key: Any) -> Array:
        y = x[key]
        self.stash(x.shape, key)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape, key = self.unstash()
        dx = self.xp.zeros(x_shape, dtype=dy.dtype)
        self.xp.add.at(dx, key, dy)
        return (dx,)


class Split(Select):
    """Splits arrays."""


class Stack(Op):
    """Stacks arrays."""

    def forward(self, *arrays: Array | bool, dim: int) -> Array:
        y = self.xp.stack(arrays, dim)
        self.stash(dim)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim = self.unstash()
        dxs = tuple(self.xp.moveaxis(dy, dim, 0))
        return tuple(dxs)


class Transpose(Op):
    """Transposes an array."""

    def forward(self, x: Array, *, dim1: int, dim2: int) -> Array:
        y = x.swapaxes(dim1, dim2)
        self.stash(dim1, dim2)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim1, dim2 = self.unstash()
        dx = dy.swapaxes(dim1, dim2)
        return (dx,)


class View(Op):
    """Reshapes an array."""

    def forward(self, x: Array, *, shape: ShapeLike) -> Array:
        y = self.xp.reshape(x, shape)
        self.stash(x.shape)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (x_shape,) = self.unstash()
        dx = dy.reshape(x_shape)
        return (dx,)


class Squeeze(View):
    """Removes singular dimensions of an array."""


class Where(Op):
    """Selects array elements based on a condition."""

    def forward(
        self,
        condition: Array,
        x1: Array,
        x2: Array,
    ) -> Array:
        y = self.xp.where(condition, x1, x2)
        self.stash(y == x1)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.unstash()
        dx1 = dy * mask
        dx2 = dy * self.xp.invert(mask)
        return None, dx1, dx2
