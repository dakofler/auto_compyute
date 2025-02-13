"""Tensor factory functions."""

from typing import Any, Optional

from .autograd import Tensor
from .backends import Device, DeviceLike, Scalar, select_device
from .dtypes import DType, int64, select_dtype

__all__ = [
    "tensor",
    "arange",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
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


def tensor(
    data: Any,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor from the given data.

    Args:
        data (Any): The underlying data of the tensor.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    if isinstance(data, Tensor):
        return data
    device, _ = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.asarray(data, dtype)
    return Tensor(data, req_grad=req_grad)


def arange(
    stop: float,
    start: float = 0,
    step: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor with values in the given range.

    Args:
        stop (float): The end value (exclusive).
        start (float, optional): The start value. Defaults to `0`.
        step (float, optional): The step size. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `int64`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.arange(start, stop, step, dtype)
    return Tensor(data, req_grad=req_grad)


def ones(
    *dims: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of ones.

    Args:
        *dims (int): The dimensions of the tensor.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.ones(dims, dtype)
    return Tensor(data, req_grad=req_grad)


def ones_like(
    x: Tensor,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of ones, matching the shape of another tensor.

    Args:
        x (Tensor): The reference tensor.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device = device if device is not None else x.device
    dtype = dtype if dtype is not None else x.dtype
    return ones(*x.shape, device=device, dtype=dtype, req_grad=req_grad)


def zeros(
    *dims: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of zeros.

    Args:
        *dims (int): The dimensions of the tensor.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.zeros(dims, dtype)
    return Tensor(data, req_grad=req_grad)


def zeros_like(
    x: Tensor,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of zeros, matching the shape of another tensor.

    Args:
        x (Tensor): The reference tensor.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device = device if device is not None else x.device
    dtype = dtype if dtype is not None else x.dtype
    return zeros(*x.shape, device=device, dtype=dtype, req_grad=req_grad)


def full(
    *dims: int,
    value: Scalar,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor filled with a specified value.

    Args:
        *dims (int): The dimensions of the tensor.
        value (Scalar): The fill value.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.full(dims, value, dtype)
    return Tensor(data, req_grad=req_grad)


def full_like(
    x: Tensor,
    value: Scalar,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor filled with a specified value, matching the shape of another tensor.

    Args:
        x (Tensor): The reference tensor.
        value (Scalar): The fill value.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device = device if device is not None else x.device
    dtype = dtype if dtype is not None else x.dtype
    return full(*x.shape, value=value, device=device, dtype=dtype, req_grad=req_grad)


def randi(
    *dims: int,
    low: int,
    high: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of random integers within a given range.

    Args:
        *dims (int): The dimensions of the tensor.
        low (int): The lower bound (inclusive).
        high (int): The upper bound (exclusive).
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType, optional): The desired data type of the tensor. Defaults to `int64`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.randint(low, high, dims, dtype)
    return Tensor(data, req_grad=req_grad)


def randi_like(
    x: Tensor,
    low: int,
    high: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of random integers within a range, matching the shape of another tensor.

    Args:
        x (Tensor): The reference tensor.
        low (int): The lower bound (inclusive).
        high (int): The upper bound (exclusive).
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType, optional): The desired data type of the tensor. Defaults to `int64`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device = device if device is not None else x.device
    dtype = dtype if dtype is not None else x.dtype
    return randi(
        *x.shape, low=low, high=high, device=device, dtype=dtype, req_grad=req_grad
    )


def randn(
    *dims: int,
    mean: float = 0,
    var: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of random values from a normal distribution.

    Args:
        *dims (int): The dimensions of the tensor.
        mean (float, optional): The mean of the distribution. Defaults to `0`.
        var (float, optional): The variance of the distribution. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.normal(mean, var, dims).astype(dtype)
    return Tensor(data, req_grad=req_grad)


def randn_like(
    x: Tensor,
    mean: float = 0,
    var: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of random values from a normal distribution, matching the shape of another
        tensor.

    Args:
        x (Tensor): The reference tensor.
        mean (float, optional): The mean of the distribution. Defaults to `0`.
        var (float, optional): The variance of the distribution. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device = device if device is not None else x.device
    dtype = dtype if dtype is not None else x.dtype
    return randn(
        *x.shape, mean=mean, var=var, device=device, dtype=dtype, req_grad=req_grad
    )


def randu(
    *dims: int,
    low: float = -1,
    high: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of random values from a uniform distribution.

    Args:
        *dims (int): The dimensions of the tensor.
        low (float, optional): The lower bound. Defaults to -1.
        high (float, optional): The upper bound. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.uniform(low, high, dims).astype(dtype)
    return Tensor(data, req_grad=req_grad)


def randu_like(
    x: Tensor,
    low: float = -1,
    high: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor of random values from a uniform distribution, matching the shape of another
        tensor.

    Args:
        x (Tensor): The reference tensor.
        mean (float, optional): The mean of the distribution. Defaults to `0`.
        var (float, optional): The variance of the distribution. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device = device if device is not None else x.device
    dtype = dtype if dtype is not None else x.dtype
    return randu(
        *x.shape, low=low, high=high, device=device, dtype=dtype, req_grad=req_grad
    )


def randperm(
    n: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
) -> Tensor:
    """Creates an tensor with a random permutation of integers from `0` to `n-1`.

    Args:
        n (int): The number of elements.
        device (DeviceLike | None, optional): The device on which to create the tensor.
            Defaults to `None`.
        dtype (DType, optional): The desired data type of the tensor. Defaults to `int64`.
        req_grad (bool, optional): Whether the tensor requires autograd tracing.
            Defaults to `False`.

    Returns:
        Tensor: The created tensor.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.permutation(n).astype(dtype)
    return Tensor(data, req_grad=req_grad)
