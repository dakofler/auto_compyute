"""Multiary autograd functions"""

from ..backends import ArrayLike, Scalar
from .function import Function


class Abs(Function):
    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = self.xp.absolute(x)
        if x_req_grad:
            self.save_to_cache(y != x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        mask = self.retrieve_from_cache()
        dx = dy
        self.xp.multiply.at(dx, mask, -1)
        return (dx,)


class Exp(Function):
    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = self.xp.exp(x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        y = self.retrieve_from_cache()
        dx = dy * y
        return (dx,)


class Pow(Function):
    def forward(self, x: ArrayLike, x_req_grad: bool, *, exp: Scalar) -> ArrayLike:
        y = x**exp
        if x_req_grad:
            self.save_to_cache(x, exp)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, exp = self.retrieve_from_cache()
        dx = dy * exp * x ** (exp - 1)
        return (dx,)


class Sqrt(Function):
    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = self.xp.sqrt(x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        y = self.retrieve_from_cache()
        dx = dy * 0.5 / y
        return (dx,)


class Tanh(Function):
    def forward(self, x: ArrayLike, x_req_grad: bool) -> ArrayLike:
        y = self.xp.tanh(x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        y = self.retrieve_from_cache()
        dx = dy * (1.0 - y * y)
        return (dx,)


class Tril(Function):
    def forward(self, x: ArrayLike, x_req_grad: bool, *, diag: int) -> ArrayLike:
        y = self.xp.tril(x, diag)
        if x_req_grad:
            self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        mask = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)


class Triu(Function):
    def forward(self, x: ArrayLike, x_req_grad: bool, *, diag: int) -> ArrayLike:
        y = self.xp.triu(x, diag)
        if x_req_grad:
            self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        mask = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)
