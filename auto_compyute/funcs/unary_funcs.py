"""Multiary autograd functions"""

from ..backends import Array, Scalar
from .function import Function


class Abs(Function):
    def forward(self, x: Array) -> Array:
        y = self.backend.absolute(x)
        self.save_to_cache(y != x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.retrieve_from_cache()
        dx = dy
        self.backend.multiply.at(dx, mask, -1)
        return (dx,)


class Exp(Function):
    def forward(self, x: Array) -> Array:
        y = self.backend.exp(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.retrieve_from_cache()
        dx = dy * y
        return (dx,)


class Pow(Function):
    def forward(self, x: Array, exp: Scalar) -> Array:
        self.save_to_cache(x, exp)
        return x**exp

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, exp = self.retrieve_from_cache()
        dx = dy * exp * x ** (exp - 1)
        return (dx,)


class Sqrt(Function):
    def forward(self, x: Array) -> Array:
        y = self.backend.sqrt(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.retrieve_from_cache()
        dx = dy * 0.5 / y
        return (dx,)


class Tanh(Function):
    def forward(self, x: Array) -> Array:
        y = self.backend.tanh(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.retrieve_from_cache()
        dx = dy * (1.0 - y * y)
        return (dx,)


class Tril(Function):
    def forward(self, x: Array, diag: int) -> Array:
        y = self.backend.tril(x, diag)
        self.save_to_cache(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)


class Triu(Function):
    def forward(self, x: Array, diag: int) -> Array:
        y = self.backend.triu(x, diag)
        self.save_to_cache(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.retrieve_from_cache()
        dx = dy * mask
        return (dx,)
