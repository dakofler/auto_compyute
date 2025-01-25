"""Binary autograd functions"""

from ..backends import Array, Scalar, get_array_backend
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
        self.ctx.x1, self.ctx.x2 = x1, x2
        return x1 * x2

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dx1 = output_grad * self.ctx.x2
        dx2 = output_grad * self.ctx.x1
        return dx1, dx2


class Div(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.ctx.x1, self.ctx.x2 = x1, x2
        return x1 / x2

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = self.ctx.x1, self.ctx.x2
        dx1 = output_grad / x2
        dx2 = output_grad * x1 * -(x2**-2)
        return dx1, dx2


class Matmul(Function):
    def forward(self, x1: Array, x2: Array) -> Array:
        self.ctx.x1, self.ctx.x2 = x1, x2
        return x1 @ x2

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dx1 = output_grad @ self.ctx.x2.swapaxes(-1, -2)
        dx2 = self.ctx.x1.swapaxes(-1, -2) @ output_grad
        return dx1, dx2


class Maximum(Function):
    def forward(self, x1: Array, x2: Array | Scalar) -> Array:
        y = get_array_backend(x1).m.maximum(x1, x2)
        self.ctx.mask = y == x1
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dx1 = output_grad * self.ctx.mask
        m = get_array_backend(output_grad).m
        dx2 = output_grad * m.invert(self.ctx.mask)
        return dx1, dx2


class Minimum(Function):
    def forward(self, x1: Array, x2: Array | Scalar) -> Array:
        y = get_array_backend(x1).m.minimum(x1, x2)
        self.ctx.mask = y == x1
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dx1 = output_grad * self.ctx.mask
        m = get_array_backend(output_grad).m
        dx2 = output_grad * m.invert(self.ctx.mask)
        return dx1, dx2
