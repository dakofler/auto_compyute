"""Tensors"""

from .autograd import Tensor
from .backends import Backend, Shape, select_backend
from .dtypes import DType, int64, select_dtype

__all__ = ["ones", "zeros", "randi", "randn", "randu"]


def get_factory_kwargs(kwargs) -> tuple[Backend, DType, bool]:
    backend = select_backend(kwargs.get("backend", None))
    dtype = select_dtype(kwargs.get("dtype", None))
    requires_grad = kwargs.get("requires_grad", False)
    return backend, dtype, requires_grad


def ones(shape: Shape, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.ones(shape, dtype)
    return Tensor(data, requires_grad=requires_grad)


def zeros(shape: Shape, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.zeros(shape, dtype)
    return Tensor(data, requires_grad=requires_grad)


def randi(shape: Shape, low: int, high: int, **factory_kwargs) -> Tensor:
    backend = select_backend(factory_kwargs.get("backend", None))
    dtype = factory_kwargs.get("dtype", int64)
    requires_grad = factory_kwargs.get("requires_grad", False)
    data = backend.m.random.randint(low, high, shape, dtype)
    return Tensor(data, requires_grad=requires_grad)


def randn(shape: Shape, mean: float = 0, var: float = 1, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.random.normal(mean, var, shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)


def randu(shape: Shape, low: float = -1, high: float = 1, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.random.uniform(low, high, shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)
