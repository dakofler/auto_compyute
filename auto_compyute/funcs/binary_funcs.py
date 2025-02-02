"""Binary autograd functions"""

from ..backends import Array
from .function import Function


class Add(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        return x1 + x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        return dy, dy


class Sub(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        return x1 - x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        return dy, -dy


class Mul(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.cache.save(x1, x2)
        return x1 * x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x1, x2 = self.cache.retrieve()
        dx1 = dy * x2
        dx2 = dy * x1
        return dx1, dx2


class Div(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.cache.save(x1, x2)
        return x1 / x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x1, x2 = self.cache.retrieve()
        dx1 = dy / x2
        dx2 = dy * x1 * -(x2**-2)
        return dx1, dx2


class Matmul(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.cache.save(x1, x2)
        return x1 @ x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x1, x2 = self.cache.retrieve()
        dx1 = dy @ x2.swapaxes(-1, -2)
        dx2 = x1.swapaxes(-1, -2) @ dy
        return dx1, dx2


class Maximum(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        y = self.m.maximum(x1, x2)
        self.cache.save(y == x1)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.cache.retrieve()
        dx1 = dy * mask
        dx2 = dy * self.m.invert(mask)
        return dx1, dx2


class Minimum(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        y = self.m.minimum(x1, x2)
        self.cache.save(y == x1)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.cache.retrieve()
        dx1 = dy * mask
        dx2 = dy * self.m.invert(mask)
        return dx1, dx2
