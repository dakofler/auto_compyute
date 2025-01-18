"""Multiary autograd functions"""

from ..backends import Array, Scalar, get_array_backend
from .function import Context, Function, unbroadcast


class Add(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | Scalar) -> Array:
        y = x1 + x2
        ctx.save(x1.shape, x2.shape if isinstance(x2, Array) else None)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1_shape, x2_shape = ctx.get()
        dx1 = unbroadcast(output_grad, x1_shape)

        if x2_shape is not None:
            dx2 = unbroadcast(output_grad, x2_shape)
            return dx1, dx2

        return (dx1,)


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | Scalar) -> Array:
        y = x1 - x2
        ctx.save(x1.shape, x2.shape if isinstance(x2, Array) else None)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1_shape, x2_shape = ctx.get()
        dx1 = unbroadcast(output_grad, x1_shape)

        if x2_shape is not None:
            dx2 = unbroadcast(-output_grad, x2_shape)
            return dx1, dx2

        return (dx1,)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | Scalar) -> Array:
        y = x1 * x2
        ctx.save(x1, x2)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = ctx.get()
        dx1_raw = output_grad * x2
        dx1 = unbroadcast(dx1_raw, x1.shape)

        if isinstance(x2, Array):
            dx2_raw = output_grad * x1
            dx2 = unbroadcast(dx2_raw, x2.shape)
            return dx1, dx2

        return (dx1,)


class Div(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | Scalar) -> Array:
        y = x1 / x2
        ctx.save(x1, x2)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = ctx.get()
        dx1_raw = output_grad / x2
        dx1 = unbroadcast(dx1_raw, x1.shape)

        if isinstance(x2, Array):
            dx2_raw = output_grad * x1 * -(x2**-2)
            dx2 = unbroadcast(dx2_raw, x2.shape)
            return dx1, dx2

        return (dx1,)


class Matmul(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array) -> Array:
        y = x1 @ x2
        ctx.save(x1, x2)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = ctx.get()
        dx1_raw = output_grad @ x2.swapaxes(-1, -2)
        dx2_raw = x1.swapaxes(-1, -2) @ output_grad
        dx1 = unbroadcast(dx1_raw, x1.shape)
        dx2 = unbroadcast(dx2_raw, x2.shape)
        return dx1, dx2


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | Scalar) -> Array:
        y = x1**x2
        ctx.save(x1, x2)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = ctx.get()
        dx1_raw = output_grad * x2 * x1 ** (x2 - 1)
        dx1 = unbroadcast(dx1_raw, x1.shape)

        if isinstance(x2, Array):
            b = get_array_backend(x2).m
            dx2_raw = output_grad * b.log(x1) * x1**x2
            dx2 = unbroadcast(dx2_raw, x2.shape)
            return dx1, dx2

        return (dx1,)


class Maximum(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | Scalar) -> Array:
        b = get_array_backend(x1).m
        y = b.maximum(x1, x2)
        ctx.save(y == x1, x1.shape, x2.shape if isinstance(x2, Array) else None)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        mask, x1_shape, x2_shape = ctx.get()
        dx1_raw = output_grad * mask
        dx1 = unbroadcast(dx1_raw, x1_shape)

        if x2_shape is not None:
            m = get_array_backend(output_grad).m
            dx2_raw = output_grad * m.invert(mask)
            dx2 = unbroadcast(dx2_raw, x2_shape)
            return dx1, dx2
        return (dx1,)


class Minimum(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | Scalar) -> Array:
        b = get_array_backend(x1).m
        y = b.minimum(x1, x2)
        ctx.save(y == x1, x1.shape, x2.shape if isinstance(x2, Array) else None)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        mask, x1_shape, x2_shape = ctx.get()
        dx1_raw = output_grad * mask
        dx1 = unbroadcast(dx1_raw, x1_shape)

        if x2_shape is not None:
            m = get_array_backend(output_grad).m
            dx2_raw = output_grad * m.invert(mask)
            dx2 = unbroadcast(dx2_raw, x2_shape)
            return dx1, dx2
        return (dx1,)
