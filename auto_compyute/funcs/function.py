"""Autograd functions"""

from abc import abstractmethod
from typing import Any

from ..backends import Array, Shape


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
        except RuntimeError as e:
            raise RuntimeError("Ran backward multiple times.") from e
        return elements if len(elements) > 1 else elements[0]


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
