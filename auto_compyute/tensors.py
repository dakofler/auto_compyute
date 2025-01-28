"""Tensors"""

from typing import Any

from .autograd import Tensor
from .devices import Device, Shape, select_device
from .dtypes import DType, float32, int64, select_dtype

__all__ = ["tensor", "arange", "ones", "zeros", "randi", "randn", "randu"]


def get_factory_kwargs(kwargs) -> tuple[Device, DType, bool]:
    device = select_device(kwargs.get("device", None))
    dtype = select_dtype(kwargs.get("dtype", float32))
    requires_grad = kwargs.get("requires_grad", False)
    return device, dtype, requires_grad


def tensor(data: Any, **factory_kwargs) -> Tensor:
    device, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    dtype = factory_kwargs.get("dtype", None)
    data = device.m.asarray(data, dtype)
    return Tensor(data, requires_grad=requires_grad)


def arange(stop: float, start: float = 0, step: float = 1, **factory_kwargs) -> Tensor:
    device, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    dtype = factory_kwargs.get("dtype", int64)
    data = device.m.arange(start, stop, step, dtype)
    return Tensor(data, requires_grad=requires_grad)


def ones(shape: Shape, **factory_kwargs) -> Tensor:
    device, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = device.m.ones(shape, dtype)
    return Tensor(data, requires_grad=requires_grad)


def zeros(shape: Shape, **factory_kwargs) -> Tensor:
    device, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = device.m.zeros(shape, dtype)
    return Tensor(data, requires_grad=requires_grad)


def randi(shape: Shape, low: int, high: int, **factory_kwargs) -> Tensor:
    device, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    dtype = factory_kwargs.get("dtype", int64)
    data = device.m.random.randint(low, high, shape, dtype)
    return Tensor(data, requires_grad=requires_grad)


def randn(shape: Shape, mean: float = 0, var: float = 1, **factory_kwargs) -> Tensor:
    device, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = device.m.random.normal(mean, var, shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)


def randu(shape: Shape, low: float = -1, high: float = 1, **factory_kwargs) -> Tensor:
    device, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = device.m.random.uniform(low, high, shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)
