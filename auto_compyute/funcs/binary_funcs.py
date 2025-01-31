"""Binary autograd functions"""

from ..backends import Array
from .function import Function


class Add(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        return x1 + x2

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        return output_grad, output_grad


class Sub(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        return x1 - x2

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        return output_grad, -output_grad


class Mul(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.ctx.save(x1, x2)
        return x1 * x2

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = self.ctx.retrieve()
        dx1 = output_grad * x2
        dx2 = output_grad * x1
        return dx1, dx2


class Div(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.ctx.save(x1, x2)
        return x1 / x2

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = self.ctx.retrieve()
        dx1 = output_grad / x2
        dx2 = output_grad * x1 * -(x2**-2)
        return dx1, dx2


class Matmul(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.ctx.save(x1, x2)
        return x1 @ x2

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = self.ctx.retrieve()
        dx1 = output_grad @ x2.swapaxes(-1, -2)
        dx2 = x1.swapaxes(-1, -2) @ output_grad
        return dx1, dx2


class Maximum(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        y = self.m.maximum(x1, x2)
        self.ctx.save(y == x1)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        mask = self.ctx.retrieve()
        dx1 = output_grad * mask
        dx2 = output_grad * self.m.invert(mask)
        return dx1, dx2


class Minimum(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        y = self.m.minimum(x1, x2)
        self.ctx.save(y == x1)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        mask = self.ctx.retrieve()
        dx1 = output_grad * mask
        dx2 = output_grad * self.m.invert(mask)
        return dx1, dx2
