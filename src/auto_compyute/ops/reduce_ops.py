"""Reduce operations."""

from typing import Optional

from auto_compyute.backends import Array
from auto_compyute.ops.op import Op


class Sum(Op):
    """Sum of array elements."""

    def forward(self, x: Array, *, dim: Optional[int | tuple[int, ...]], keepdims: bool) -> Array:
        y = x.sum(dim, keepdims=keepdims)
        self.save_to_cache(x.shape, dim, keepdims)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape, dim, keepdims = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = self.xp.broadcast_to(dy, x_shape)
        return (dx,)


class Mean(Op):
    """Mean of array elements."""

    def forward(self, x: Array, *, dim: Optional[int | tuple[int, ...]], keepdims: bool) -> Array:
        y = x.mean(dim, keepdims=keepdims)
        self.save_to_cache(x.shape, dim, keepdims, x.size / y.size)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x_shape, dim, keepdims, size = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = self.xp.broadcast_to(dy / size, x_shape)
        return (dx,)


class Var(Op):
    """Variance of array elements."""

    def forward(
        self,
        x: Array,
        *,
        dim: Optional[int | tuple[int, ...]],
        ddof: int,
        keepdims: bool,
    ) -> Array:
        y = x.var(dim, ddof=ddof, keepdims=keepdims)
        self.save_to_cache(x, dim, x.size / y.size - ddof)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, dim, n = self.retrieve_from_cache()
        dx = dy * 2.0 * (x - x.mean(dim, keepdims=True)) / n
        return (dx,)


class Std(Op):
    """Standard deviation of array elements."""

    def forward(
        self,
        x: Array,
        *,
        dim: Optional[int | tuple[int, ...]],
        ddof: int,
        keepdims: bool,
    ) -> Array:
        y = x.std(dim, ddof=ddof, keepdims=keepdims)
        self.save_to_cache(x, dim, ddof, y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, dim, ddof, y = self.retrieve_from_cache()
        n = x.size / y.size - ddof
        dx = dy * (x - x.mean(dim, keepdims=True)) / (n * y)
        return (dx,)


class Max(Op):
    """Maximum of array elements."""

    def forward(self, x: Array, *, dim: Optional[int], keepdims: bool) -> Array:
        y = x.max(dim, keepdims=True)
        self.save_to_cache(dim, keepdims, x == y)
        return y if keepdims else y.squeeze()

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, keepdims, mask = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = mask * dy / mask.sum(dim, dtype=dy.dtype, keepdims=True)
        return (dx,)


class Min(Op):
    """Minimum of array elements."""

    def forward(self, x: Array, *, dim: Optional[int], keepdims: bool) -> Array:
        y = x.min(dim, keepdims=True)
        self.save_to_cache(dim, keepdims, x == y)
        return y if keepdims else y.squeeze()

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dim, keepdims, mask = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = mask * dy / mask.sum(dim, dtype=dy.dtype, keepdims=True)
        return (dx,)
