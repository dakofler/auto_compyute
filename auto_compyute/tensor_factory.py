"""Tensor factory functions"""

from typing import Any, Optional

from .autograd import Tensor
from .backends import Device, DeviceLike, Scalar, select_device
from .dtypes import DType, int64, select_dtype

__all__ = [
    "tensor",
    "arange",
    "ones",
    "zeros",
    "full",
    "randi",
    "randn",
    "randu",
    "randperm",
]


def _parse_factory_kwargs(
    device: Optional[DeviceLike], dtype: Optional[DType]
) -> tuple[Device, DType]:
    device = select_device(device)
    dtype = select_dtype(dtype)
    return device, dtype


def tensor(
    data: Any,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    requires_grad: bool = False,
) -> Tensor:
    device, _ = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.asarray(data, dtype)
    return Tensor(data, requires_grad=requires_grad)


def arange(
    stop: float,
    start: float = 0,
    step: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    requires_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.arange(start, stop, step, dtype)
    return Tensor(data, requires_grad=requires_grad)


def ones(
    *dims: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    requires_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.ones(dims, dtype)
    return Tensor(data, requires_grad=requires_grad)


def zeros(
    *dims: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    requires_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.zeros(dims, dtype)
    return Tensor(data, requires_grad=requires_grad)


def full(
    *dims: int,
    value: Scalar,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    requires_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.full(dims, value, dtype)
    return Tensor(data, requires_grad=requires_grad)


def randi(
    *dims: int,
    low: int,
    high: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    requires_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.random.randint(low, high, dims, dtype)
    return Tensor(data, requires_grad=requires_grad)


def randn(
    *dims: int,
    mean: float = 0,
    var: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    requires_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.random.normal(mean, var, dims).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)


def randu(
    *dims: int,
    low: float = -1,
    high: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    requires_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.random.uniform(low, high, dims).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)


def randperm(
    n: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    requires_grad: bool = False,
):
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.backend.random.permutation(n).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)
