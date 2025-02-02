"""Backend devices"""

import re
from typing import Any, Optional, TypeAlias

import cupy  # type: ignore
import numpy  # type: ignore

__all__ = ["Device"]

Array: TypeAlias = cupy.ndarray | numpy.ndarray
Scalar: TypeAlias = int | float
Shape = tuple[int, ...]
Dim = int | tuple[int, ...]

MAX_LINE_WIDTH = 200
PRECISION = 4
FLOATMODE = "maxprec_equal"

numpy.set_printoptions(
    precision=PRECISION, linewidth=MAX_LINE_WIDTH, floatmode=FLOATMODE
)

cupy.set_printoptions(
    precision=PRECISION, linewidth=MAX_LINE_WIDTH, floatmode=FLOATMODE
)


def gpu_available():
    return cupy.cuda.is_available()


def set_random_seed(seed: int):
    numpy.random.seed(seed)
    if gpu_available():
        cupy.random.seed(seed)


class Device:
    def __init__(self, device_type: str):
        device_type, device_id = _get_type_and_id(device_type)
        self.device_type = device_type
        self.device_id = device_id
        self.backend = numpy if device_type == "cpu" else cupy

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Device)
            and other.device_type == self.device_type
            and other.device_id == self.device_id
        )

    def __repr__(self):
        id_suffix = f":{self.device_id}" if self.device_type == "cuda" else ""
        return f"Device({self.device_type}{id_suffix})"

    def __enter__(self) -> None:
        if self.device_type == "cpu":
            return None
        return cupy.cuda.Device(self.device_id).__enter__()

    def __exit__(self, *args: Any) -> None:
        if self.device_type == "cpu":
            return None
        return cupy.cuda.Device(self.device_id).__exit__(*args)


def _get_type_and_id(device_type: str) -> tuple[str, Optional[int]]:
    match = re.match(r"(?P<type>cpu|cuda)(?::(?P<id>\d+))?", device_type)
    if match:
        device_type = match.group("type")
        if device_type == "cuda":
            assert gpu_available()
        device_id = match.group("id")
        return (device_type, None if device_id is None else int(device_id))
    raise ValueError(f"Unknown device: {device_type}")


DeviceLike: TypeAlias = Device | str


def get_array_device(x: Array) -> Device:
    return Device("cuda:0") if isinstance(x, cupy.ndarray) else Device("cpu")


def select_device(device: Optional[DeviceLike]) -> Device:
    if isinstance(device, Device):
        return device
    return Device(device or "cpu")


def move_to_device(data: Array, device: Device) -> Array:
    if device == Device("cpu"):
        return cupy.asnumpy(data)
    return cupy.asarray(data)


def array_to_string(data: Array, prefix: str) -> str:
    device = get_array_device(data)
    return device.backend.array2string(
        data,
        max_line_width=MAX_LINE_WIDTH,
        precision=PRECISION,
        separator=", ",
        prefix=prefix,
        floatmode=FLOATMODE,
    )
