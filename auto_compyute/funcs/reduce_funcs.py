"""Reduce autograd functions"""

from typing import Optional

from ..backends import ArrayLike
from .function import Function


class Sum(Function):
    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        *,
        dim: Optional[int | tuple[int, ...]],
        keepdims: bool,
    ) -> ArrayLike:
        y = x.sum(dim, keepdims=keepdims)
        if x_req_grad:
            self.save_to_cache(x.shape)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape = self.retrieve_from_cache()
        dx = self.xp.broadcast_to(dy, x_shape)
        return (dx,)


class Mean(Function):
    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        *,
        dim: Optional[int | tuple[int, ...]],
        keepdims: bool,
    ) -> ArrayLike:
        y = x.mean(dim, keepdims=keepdims)
        if x_req_grad:
            self.save_to_cache(x.shape, x.size / y.size)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape, size = self.retrieve_from_cache()
        dx = self.xp.broadcast_to(dy / size, x_shape)
        return (dx,)


class Var(Function):
    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        *,
        dim: Optional[int | tuple[int, ...]],
        ddof: int,
        keepdims: bool,
    ) -> ArrayLike:
        y = x.var(dim, ddof=ddof, keepdims=keepdims)
        if x_req_grad:
            self.save_to_cache(x, dim, x.size / y.size - ddof)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, dim, n = self.retrieve_from_cache()
        dx = dy * 2.0 * (x - x.mean(dim, keepdims=True)) / n
        return (dx,)


class Std(Function):
    def forward(
        self,
        x: ArrayLike,
        x_req_grad: bool,
        *,
        dim: Optional[int | tuple[int, ...]],
        ddof: int,
        keepdims: bool,
    ) -> ArrayLike:
        y = x.std(dim, ddof=ddof, keepdims=keepdims)
        if x_req_grad:
            self.save_to_cache(x, dim, ddof, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, dim, ddof, y = self.retrieve_from_cache()
        n = x.size / y.size - ddof
        dx = dy * (x - x.mean(dim, keepdims=True)) / (n * y)
        return (dx,)


class Max(Function):
    def forward(
        self, x: ArrayLike, x_req_grad: bool, *, dim: Optional[int], keepdims: bool
    ) -> ArrayLike:
        y = x.max(dim, keepdims=True)
        if x_req_grad:
            self.save_to_cache(dim, keepdims, x == y)
        return y if keepdims else y.squeeze()

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, keepdims, mask = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = mask * dy / mask.sum(dim, dtype=dy.dtype, keepdims=True)
        return (dx,)


class Min(Function):
    def forward(
        self, x: ArrayLike, x_req_grad: bool, *, dim: Optional[int], keepdims: bool
    ) -> ArrayLike:
        y = x.min(dim, keepdims=True)
        if x_req_grad:
            self.save_to_cache(dim, keepdims, x == y)
        return y if keepdims else y.squeeze()

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, keepdims, mask = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = mask * dy / mask.sum(dim, dtype=dy.dtype, keepdims=True)
        return (dx,)
