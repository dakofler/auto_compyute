"""Autograd functions"""

from abc import abstractmethod
from typing import Any, Optional

from .backends import Array, Shape, get_array_backend


def get_shape_diff(shape1: Shape, shape2: Shape) -> Shape:
    return tuple(i for i in range(len(shape1)) if shape1[i] != shape2[i])


def unbroadcast(grad: Array, target_shape: Shape) -> Array:
    if grad.shape != target_shape:
        target_ndim = len(target_shape)

        if grad.ndim == target_ndim:
            axis = get_shape_diff(grad.shape, target_shape)
            grad = grad.sum(axis, keepdims=True)
        else:
            data_shape = (1,) * (grad.ndim - target_ndim) + target_shape
            axis = get_shape_diff(grad.shape, data_shape)
            grad = grad.sum(axis=axis)

        grad = grad.reshape(target_shape)

    return grad


class Function:
    @staticmethod
    @abstractmethod
    def forward(ctx, *args, **kwargs) -> Array:
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx, output_grad) -> tuple[Array, ...]:
        pass


class Context:
    def __init__(self) -> None:
        self.elements: list[Any] = []

    def save(self, *elements):
        self.elements.append(elements)

    def get(self):
        try:
            elements = self.elements.pop()
        except:
            raise ValueError("Ran backward multiple times.")
        return elements if len(elements) > 1 else elements[0]


class Add(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | int | float) -> Array:
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


class Subtract(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | int | float) -> Array:
        y = x1 - x2
        ctx.save(x1.shape, x2.shape if isinstance(x2, Array) else None)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1_shape, x2_shape = ctx.get()
        dx1 = unbroadcast(output_grad, x1_shape)
        if x2_shape is not None:
            dx2 = unbroadcast(output_grad, x2_shape)
            return dx1, -dx2
        return (dx1,)


class Multiply(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | int | float) -> Array:
        y = x1 * x2
        ctx.save(x1, x2)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = ctx.get()
        dx1 = unbroadcast(output_grad * x2, x1.shape)
        dx2 = unbroadcast(output_grad * x1, x2.shape)
        return dx1, dx2


class Divide(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | int | float) -> Array:
        y = x1 / x2
        ctx.save(x1, x2)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = ctx.get()
        dx1 = unbroadcast(output_grad / x2, x1.shape)
        dx2 = unbroadcast(output_grad * x1 * -(x2**-2), x2.shape)
        return dx1, dx2


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: int) -> Array:
        y = x1**x2
        ctx.save(x1, x2)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x1, x2 = ctx.get()
        dx1 = output_grad * x2 * x1 ** (x2 - 1)
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
        dx1 = unbroadcast(output_grad @ x2.swapaxes(-1, -2), x1.shape)
        dx2 = unbroadcast(x1.swapaxes(-1, -2) @ output_grad, x2.shape)
        return dx1, dx2


class Maximum(Function):
    @staticmethod
    def forward(ctx: Context, x1: Array, x2: Array | int | float) -> Array:
        b = get_array_backend(x1).m
        y = b.maximum(x1, x2)
        ctx.save(y == x1, isinstance(x2, Array))
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        mask, x2_is_array = ctx.get()
        m = get_array_backend(output_grad).m
        dx1 = output_grad * mask
        if x2_is_array:
            dx2 = output_grad * m.invert(mask)
            return dx1, dx2
        return (dx1,)


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, x: Array, dims) -> Array:
        y = x.swapaxes(*dims)
        ctx.save(dims)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        dims = ctx.get()
        dx = output_grad.swapaxes(*dims)
        return (dx,)


class Sum(Function):
    @staticmethod
    def forward(
        ctx: Context, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        y = x.sum(dim, keepdims=keepdims)
        ctx.save(x.shape)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x_shape = ctx.get()
        b = get_array_backend(output_grad).m
        dx = b.broadcast_to(output_grad, x_shape)
        return (dx,)


class Mean(Function):
    @staticmethod
    def forward(
        ctx: Context, x: Array, dim: Optional[int | tuple[int, ...]], keepdims: bool
    ) -> Array:
        y = x.mean(dim, keepdims=keepdims)
        ctx.save(x.shape)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        x_shape = ctx.get()
        b = get_array_backend(output_grad).m
        dx = b.broadcast_to(output_grad / b.prod(x_shape), x_shape)
        return (dx,)


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, x: Array) -> Array:
        b = get_array_backend(x).m
        y = b.tanh(x)
        ctx.save(y)
        return y

    @staticmethod
    def backward(ctx: Context, output_grad: Array) -> tuple[Array, ...]:
        y = ctx.get()
        dx = output_grad * (1 - y**2)
        return (dx,)
