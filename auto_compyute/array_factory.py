"""Tensor factory functions"""

from typing import Any, Optional

from .autograd import Array
from .backends import Device, DeviceLike, Scalar, select_device
from .dtypes import DType, int64, select_dtype

__all__ = [
    "array",
    "arange",
    "ones",
    "ones_like",
    "zeros",
    "ones_like",
    "full",
    "full_like",
    "randi",
    "randi_like",
    "randn",
    "randn_like",
    "randu",
    "randu_like",
    "randperm",
]


def _parse_factory_kwargs(
    device: Optional[DeviceLike], dtype: Optional[DType]
) -> tuple[Device, DType]:
    device = select_device(device)
    dtype = select_dtype(dtype)
    return device, dtype


def array(
    data: Any,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    device, _ = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.asarray(data, dtype)
    return Array(data, req_grad=req_grad)


def arange(
    stop: float,
    start: float = 0,
    step: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
) -> Array:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.arange(start, stop, step, dtype)
    return Array(data, req_grad=req_grad)


def ones(
    *dims: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.ones(dims, dtype)
    return Array(data, req_grad=req_grad)


def ones_like(x: Array, req_grad: bool = False) -> Array:
    return ones(*x.shape, device=x.device, dtype=x.dtype, req_grad=req_grad)


def zeros(
    *dims: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.zeros(dims, dtype)
    return Array(data, req_grad=req_grad)


def zeros_like(x: Array, req_grad: bool = False) -> Array:
    return zeros(*x.shape, device=x.device, dtype=x.dtype, req_grad=req_grad)


def full(
    *dims: int,
    value: Scalar,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.full(dims, value, dtype)
    return Array(data, req_grad=req_grad)


def full_like(x: Array, value: Scalar, req_grad: bool = False) -> Array:
    return full(
        *x.shape, value=value, device=x.device, dtype=x.dtype, req_grad=req_grad
    )


def randi(
    *dims: int,
    low: int,
    high: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
) -> Array:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.randint(low, high, dims, dtype)
    return Array(data, req_grad=req_grad)


def randi_like(
    x: Array,
    low: int,
    high: int,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
) -> Array:
    return randi(
        *x.shape, low=low, high=high, device=x.device, dtype=dtype, req_grad=req_grad
    )


def randn(
    *dims: int,
    mean: float = 0,
    var: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.normal(mean, var, dims).astype(dtype)
    return Array(data, req_grad=req_grad)


def randn_like(
    x: Array,
    mean: float = 0,
    var: float = 1,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    return randn(
        *x.shape, mean=mean, var=var, device=x.device, dtype=dtype, req_grad=req_grad
    )


def randu(
    *dims: int,
    low: float = -1,
    high: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.uniform(low, high, dims).astype(dtype)
    return Array(data, req_grad=req_grad)


def randu_like(
    x: Array,
    low: float = -1,
    high: float = 1,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    return randu(
        *x.shape, low=low, high=high, device=x.device, dtype=dtype, req_grad=req_grad
    )


def randperm(
    n: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
):
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.permutation(n).astype(dtype)
    return Array(data, req_grad=req_grad)
