"""Unary operations."""

from auto_compyute.backends import Array, Scalar
from auto_compyute.ops.op import Op


class Abs(Op):
    """Element-wise absolute value."""

    def forward(self, x: Array) -> Array:
        y = self.xp.absolute(x)
        self.stash(y != x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (mask,) = self.unstash()
        dx = dy
        self.xp.multiply.at(dx, mask, -1)
        return (dx,)


class Exp(Op):
    """Element-wise exponential."""

    def forward(self, x: Array) -> Array:
        y = self.xp.exp(x)
        self.stash(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (y,) = self.unstash()
        dx = dy * y
        return (dx,)


class Log(Op):
    """Element-wise natural logarithm."""

    def forward(self, x: Array) -> Array:
        y = self.xp.log(x)
        self.stash(x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x = self.unstash()
        dx = dy / x
        return (dx,)


class Pow(Op):
    """Element-wise power."""

    def forward(self, x: Array, *, exp: Scalar) -> Array:
        y = x**exp
        self.stash(x, exp)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x, exp = self.unstash()
        dx = dy * exp * x ** (exp - 1)
        return (dx,)


class Sqrt(Op):
    """Element-wise square root."""

    def forward(self, x: Array) -> Array:
        y = self.xp.sqrt(x)
        self.stash(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (y,) = self.unstash()
        dx = dy * 0.5 / y
        return (dx,)


class Tanh(Op):
    """Element-wise hyperbolic tangent."""

    def forward(self, x: Array) -> Array:
        y = self.xp.tanh(x)
        self.stash(y)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (y,) = self.unstash()
        dx = dy * (1 - y * y)
        return (dx,)


class Tril(Op):
    """Sets lower diagonal elements to `0`."""

    def forward(self, x: Array, *, diag: int) -> Array:
        y = self.xp.tril(x, diag)
        self.stash(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (mask,) = self.unstash()
        dx = dy * mask
        return (dx,)


class Triu(Op):
    """Sets upper diagonal elements to `0`."""

    def forward(self, x: Array, *, diag: int) -> Array:
        y = self.xp.triu(x, diag)
        self.stash(y == x)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        (mask,) = self.unstash()
        dx = dy * mask
        return (dx,)
