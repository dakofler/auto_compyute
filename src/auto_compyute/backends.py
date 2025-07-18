"""Computation backends."""

import re
from typing import Any, Optional, TypeAlias

import numpy  # type: ignore

__all__ = ["Device", "get_available_devices", "gpu_available", "set_random_seed"]


# -------------------------------------------------------------------------------------
# GENERAL
# -------------------------------------------------------------------------------------

Scalar: TypeAlias = int | float
Dim: TypeAlias = int | tuple[int, ...]


class Shape(tuple):
    """Array shape as a tuple of integers."""

    def __repr__(self) -> str:
        return f"shape({super().__repr__().replace('(', '').replace(')', '')})"


ShapeLike: TypeAlias = Shape | tuple[int, ...]


# -------------------------------------------------------------------------------------
# BACKENDS
# -------------------------------------------------------------------------------------


MAX_LINE_WIDTH = 200
PREC = 4
FLOATMODE = "maxprec_equal"

CPU_BACKEND = numpy
CPU_BACKEND.set_printoptions(precision=PREC, linewidth=MAX_LINE_WIDTH, floatmode=FLOATMODE)

try:
    import cupy  # type: ignore

    GPU_BACKEND = cupy  # type: ignore
    GPU_BACKEND.set_printoptions(precision=PREC, linewidth=MAX_LINE_WIDTH, floatmode=FLOATMODE)

except ImportError:
    GPU_BACKEND = None

Array: TypeAlias = numpy.ndarray


def gpu_available() -> bool:
    """Returns `True` if at least one GPU device is available.

    Returns:
        bool: `True` if at least one CUDA-compatible GPU is available, otherwise `False`.
    """
    return GPU_BACKEND is not None and GPU_BACKEND.cuda.is_available()


def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducibility on all devices.

    Args:
        seed (int): The seed value to set.
    """
    CPU_BACKEND.random.seed(seed)
    if gpu_available():
        GPU_BACKEND.random.seed(seed)


def array_to_string(data: Array, prefix: str) -> str:
    """Converts an array to a formatted string.

    Args:
        data (ArrayLike): The array to convert.
        prefix (str): A prefix for formatting the output.

    Returns:
        str: A string representation of the array.
    """
    device = get_array_device(data)
    return device.xp.array2string(
        data,
        max_line_width=MAX_LINE_WIDTH,
        precision=PREC,
        separator=", ",
        prefix=prefix,
        floatmode=FLOATMODE,
    )


# -------------------------------------------------------------------------------------
# DEVICE
# -------------------------------------------------------------------------------------


def _get_type_and_id(device_type: str) -> tuple[str, Optional[int]]:
    match = re.match(r"(?P<type>cpu|cuda)(?::(?P<id>\d+))?", device_type)
    if match:
        device_type = match.group("type")
        if device_type == "cuda":
            assert gpu_available(), "GPUs are not available."
        device_id = match.group("id")
        return (device_type, None if device_id is None else int(device_id))
    raise ValueError(f"Unknown device: {device_type}")


class Device:
    """Represents a device to store data and perform computations.

    Args:
        dev_type (str): The type and optionally the id of device (e.g. "cpu" or "cuda:0").
    """

    def __init__(self, device_type: str) -> None:
        device_type, dev_id = _get_type_and_id(device_type)
        self.device_type = device_type
        self.dev_id = dev_id
        self.xp = CPU_BACKEND if device_type == "cpu" else GPU_BACKEND

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Device)
            and other.device_type == self.device_type
            and other.dev_id == self.dev_id
        )

    def __repr__(self) -> str:
        id_suffix = f":{self.dev_id}" if self.device_type == "cuda" else ""
        return f"device('{self.device_type}{id_suffix}')"

    def __str__(self) -> str:
        id_suffix = f":{self.dev_id}" if self.device_type == "cuda" else ""
        return f"{self.device_type}{id_suffix}"

    def __enter__(self) -> None:
        if self.device_type == "cpu":
            return None
        return GPU_BACKEND.cuda.Device(self.dev_id).__enter__()

    def __exit__(self, *args: Any) -> None:
        if self.device_type == "cpu":
            return None
        return GPU_BACKEND.cuda.Device(self.dev_id).__exit__(*args)


DeviceLike: TypeAlias = Device | str


def get_available_devices() -> list[str]:
    """Returns a list of available devices, including CPU and CUDA GPUs.

    Returns:
        list[str]: A list of device names (e.g., ["cpu", "cuda:0", "cuda:1", ...]).
    """
    devices = ["cpu"]
    if gpu_available():
        num_gpu_devices = GPU_BACKEND.cuda.runtime.getDeviceCount()
        gpu_devices = [f"cuda:{i}" for i in range(num_gpu_devices)]
        devices.extend(gpu_devices)
    return devices


def get_array_device(x: Array) -> Device:
    """Determines the device of the given array.

    Args:
        x (ArrayLike): The input array.

    Returns:
        Device: A Device instance representing either CPU or CUDA.
    """
    return Device("cpu") if "numpy" in str(type(x)) else Device(f"cuda:{x.device.id}")


def select_device(device: Optional[DeviceLike]) -> Device:
    """Selects a device, defaulting to CPU if none is provided.

    Args:
        device (DeviceLike | None): The device to select.

    Returns:
        Device: A Device instance corresponding to the selected device.
    """
    if isinstance(device, Device):
        return device
    return Device(device or "cpu")


def parse_device(device: DeviceLike) -> Device:
    """Parses a device-like input into a Device instance.

    Args:
        device (DeviceLike): The device to parse.

    Returns:
        Device: A parsed Device instance.
    """
    return device if isinstance(device, Device) else Device(device)


def move_to_device(data: Array, device: Device) -> Array:
    """Moves an array to the specified device.

    Args:
        data (ArrayLike): The array to move.
        device (Device): The target device.

    Returns:
        ArrayLike: The array moved to the specified device.
    """
    assert gpu_available(), "GPUs are not available."
    if device == Device("cpu"):
        return GPU_BACKEND.asnumpy(data)
    return cupy.asarray(data)
