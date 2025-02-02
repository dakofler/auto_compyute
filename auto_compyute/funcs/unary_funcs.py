"""Multiary autograd functions"""

from ..backends import Array, Scalar
from .function import Function


class Abs(Function):
    def forward(self, x: Array) -> Array:
        y = self.m.absolute(x)
        self.cache.save(y != x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.cache.retrieve()
        dx = dy
        self.m.multiply.at(dx, mask, -1)
        return (dx,)


class Exp(Function):
    def forward(self, x: Array) -> Array:
        y = self.m.exp(x)
        self.cache.save(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.cache.retrieve()
        dx = dy * y
        return (dx,)


class Pow(Function):
    def forward(self, x: Array, exp: Scalar) -> Array:
        self.cache.save(x, exp)
        return x**exp

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, exp = self.cache.retrieve()
        dx = dy * exp * x ** (exp - 1)
        return (dx,)


class Sqrt(Function):
    def forward(self, x: Array) -> Array:
        y = self.m.sqrt(x)
        self.cache.save(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.cache.retrieve()
        dx = dy * 0.5 / y
        return (dx,)


class Tanh(Function):
    def forward(self, x: Array) -> Array:
        y = self.m.tanh(x)
        self.cache.save(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        y = self.cache.retrieve()
        dx = dy * (1 - y**2)
        return (dx,)


class Tril(Function):
    def forward(self, x: Array, diag: int) -> Array:
        y = self.m.tril(x, diag)
        self.cache.save(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.cache.retrieve()
        dx = dy * mask
        return (dx,)


class Triu(Function):
    def forward(self, x: Array, diag: int) -> Array:
        y = self.m.triu(x, diag)
        self.cache.save(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.cache.retrieve()
        dx = dy * mask
        return (dx,)
