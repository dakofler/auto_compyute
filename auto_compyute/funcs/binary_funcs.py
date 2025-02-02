"""Binary autograd functions"""

from ..backends import Array
from .function import Function


class Add(Function):
    def forward(self, x1: Array, x2: Array, x2_requires_grad: bool) -> Array:
        self.cache.save(x2_requires_grad)
        return x1 + x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x2_requires_grad = self.cache.retrieve()
        return (dy,) if x2_requires_grad else (dy, dy)


class Sub(Function):
    def forward(self, x1: Array, x2: Array, x2_requires_grad: bool) -> Array:
        self.cache.save(x2_requires_grad)
        return x1 - x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x2_requires_grad = self.cache.retrieve()
        return (dy,) if x2_requires_grad else (dy, -dy)


class Mul(Function):
    def forward(self, x1: Array, x2: Array, x2_requires_grad: bool) -> Array:
        x2_requires_grad = x2_requires_grad
        self.cache.save(x2_requires_grad, (None if x2_requires_grad else x1), x2)
        return x1 * x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x2_requires_grad, x1, x2 = self.cache.retrieve()
        dx1 = dy * x2
        if x2_requires_grad:
            return (dx1,)
        dx2 = dy * x1
        return (dx1, dx2)


class Div(Function):
    def forward(self, x1: Array, x2: Array, x2_requires_grad: bool) -> Array:
        x2_requires_grad = x2_requires_grad
        self.cache.save(x2_requires_grad, (None if x2_requires_grad else x1), x2)
        return x1 / x2

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x2_requires_grad, x1, x2 = self.cache.retrieve()
        dx1 = dy / x2
        if x2_requires_grad:
            return (dx1,)
        dx2 = dy * x1 * -(x2**-2)
        return (dx1, dx2)


class Matmul(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.cache.save(x1, x2)
        return self.m.einsum("...ij,...jk->...ik", x1, x2)

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x1, x2 = self.cache.retrieve()
        dx1 = self.m.einsum("...ik,...jk->...ij", dy, x2)
        dx2 = self.m.einsum("...ij,...ik->...jk", x1, dy)
        return dx1, dx2


class Maximum(Function):
    def forward(self, x1: Array, x2: Array, x2_requires_grad: bool) -> Array:
        y = self.m.maximum(x1, x2)
        self.cache.save(x2_requires_grad, y == x1)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x2_requires_grad, mask = self.cache.retrieve()
        dx1 = dy * mask
        if x2_requires_grad:
            return (dx1,)
        dx2 = dy * self.m.invert(mask)
        return dx1, dx2


class Minimum(Function):
    def forward(self, x1: Array, x2: Array, x2_requires_grad: bool) -> Array:
        y = self.m.minimum(x1, x2)
        self.cache.save(x2_requires_grad, y == x1)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x2_requires_grad, mask = self.cache.retrieve()
        dx1 = dy * mask
        if x2_requires_grad:
            return (dx1,)
        dx2 = dy * self.m.invert(mask)
        return dx1, dx2
