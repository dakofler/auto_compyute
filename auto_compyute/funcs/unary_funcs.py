"""Multiary autograd functions"""

from ..backends import Array, Scalar
from .function import Function


class Abs(Function):
    def forward(self, x: Array, x_req_grad: bool) -> Array:
        y = self.xp.absolute(x)
        if x_req_grad:
            self.save_to_cache(y != x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.retrieve_from_cache()
        dx = dy
        self.xp.multiply.at(dx, mask, -1)
        return (dx,)


class Exp(Function):
    def forward(self, x: Array, x_req_grad: bool) -> Array:
        y = self.xp.exp(x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.retrieve_from_cache()
        dx = dy * y
        return (dx,)


class Pow(Function):
    def forward(self, x: Array, x_req_grad: bool, *, exp: Scalar) -> Array:
        y = x**exp
        if x_req_grad:
            self.save_to_cache(x, exp)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, exp = self.retrieve_from_cache()
        dx = dy * exp * x ** (exp - 1)
        return (dx,)


class Sqrt(Function):
    def forward(self, x: Array, x_req_grad: bool) -> Array:
        y = self.xp.sqrt(x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.retrieve_from_cache()
        dx = dy * 0.5 / y
        return (dx,)


class Tanh(Function):
    def forward(self, x: Array, x_req_grad: bool) -> Array:
        y = self.xp.tanh(x)
        if x_req_grad:
            self.save_to_cache(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.retrieve_from_cache()
        dx = dy * (1.0 - y * y)
        return (dx,)


class Tril(Function):
    def forward(self, x: Array, x_req_grad: bool, *, diag: int) -> Array:
        y = self.xp.tril(x, diag)
        if x_req_grad:
            self.save_to_cache(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)


class Triu(Function):
    def forward(self, x: Array, x_req_grad: bool, *, diag: int) -> Array:
        y = self.xp.triu(x, diag)
        if x_req_grad:
            self.save_to_cache(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)
