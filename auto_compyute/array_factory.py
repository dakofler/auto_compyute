"""Array factory functions."""

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


def array(
    data: Any,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    """Creates an array from the given data.

    Args:
        data (Any): Input data to be converted into an Array.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    if isinstance(data, Array):
        return data
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
    """Creates an array with values in the given range.

    Args:
        stop (float): The end value (exclusive).
        start (float, optional): The start value. Defaults to `0`.
        step (float, optional): The step size. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `int64`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
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
    """Creates an array of ones.

    Args:
        *dims (int): The dimensions of the array.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.ones(dims, dtype)
    return Array(data, req_grad=req_grad)


def ones_like(
    x: Array,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    """Creates an array of ones, matching the shape of another array.

    Args:
        x (Array): The reference array.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    device = device if device is not None else x.device
    dtype = dtype if dtype is not None else x.dtype
    return ones(*x.shape, device=device, dtype=dtype, req_grad=req_grad)


def zeros(
    *dims: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    """Creates an array of zeros.

    Args:
        *dims (int): The dimensions of the array.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.zeros(dims, dtype)
    return Array(data, req_grad=req_grad)


def zeros_like(
    x: Array,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    """Creates an array of zeros, matching the shape of another array.

    Args:
        x (Array): The reference array.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
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
) -> Array:
    """Creates an array filled with a specified value.

    Args:
        *dims (int): The dimensions of the array.
        value (Scalar): The fill value.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.full(dims, value, dtype)
    return Array(data, req_grad=req_grad)


def full_like(
    x: Array,
    value: Scalar,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    """Creates an array filled with a specified value, matching the shape of another array.

    Args:
        x (Array): The reference array.
        value (Scalar): The fill value.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
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
) -> Array:
    """Creates an array of random integers within a given range.

    Args:
        *dims (int): The dimensions of the array.
        low (int): The lower bound (inclusive).
        high (int): The upper bound (exclusive).
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType, optional): The desired data type of the array. Defaults to `int64`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.randint(low, high, dims, dtype)
    return Array(data, req_grad=req_grad)


def randi_like(
    x: Array,
    low: int,
    high: int,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = int64,
    req_grad: bool = False,
) -> Array:
    """Creates an array of random integers within a range, matching the shape of another array.

    Args:
        x (Array): The reference array.
        low (int): The lower bound (inclusive).
        high (int): The upper bound (exclusive).
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType, optional): The desired data type of the array. Defaults to `int64`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
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
) -> Array:
    """Creates an array of random values from a normal distribution.

    Args:
        *dims (int): The dimensions of the array.
        mean (float, optional): The mean of the distribution. Defaults to `0`.
        var (float, optional): The variance of the distribution. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.normal(mean, var, dims).astype(dtype)
    return Array(data, req_grad=req_grad)


def randn_like(
    x: Array,
    mean: float = 0,
    var: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    """Creates an array of random values from a normal distribution, matching the shape of another
        array.

    Args:
        x (Array): The reference array.
        mean (float, optional): The mean of the distribution. Defaults to `0`.
        var (float, optional): The variance of the distribution. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
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
) -> Array:
    """Creates an array of random values from a uniform distribution.

    Args:
        *dims (int): The dimensions of the array.
        low (float, optional): The lower bound. Defaults to -1.
        high (float, optional): The upper bound. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.uniform(low, high, dims).astype(dtype)
    return Array(data, req_grad=req_grad)


def randu_like(
    x: Array,
    low: float = -1,
    high: float = 1,
    device: Optional[DeviceLike] = None,
    dtype: Optional[DType] = None,
    req_grad: bool = False,
) -> Array:
    """Creates an array of random values from a uniform distribution, matching the shape of another
        array.

    Args:
        x (Array): The reference array.
        mean (float, optional): The mean of the distribution. Defaults to `0`.
        var (float, optional): The variance of the distribution. Defaults to `1`.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType | None, optional): The desired data type of the array. Defaults to `None`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
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
) -> Array:
    """Creates an array with a random permutation of integers from `0` to `n-1`.

    Args:
        n (int): The number of elements.
        device (DeviceLike | None, optional): The device on which to create the array.
            Defaults to `None`.
        dtype (DType, optional): The desired data type of the array. Defaults to `int64`.
        req_grad (bool, optional): Whether the array requires gradient tracking.
            Defaults to `False`.

    Returns:
        Array: The created array.
    """
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.permutation(n).astype(dtype)
    return Array(data, req_grad=req_grad)
