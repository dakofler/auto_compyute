"""Backend devices"""

import re
from typing import Any, Optional, TypeAlias

import cupy  # type: ignore
import numpy  # type: ignore

__all__ = ["Device"]

ArrayLike: TypeAlias = cupy.ndarray | numpy.ndarray
Scalar: TypeAlias = int | float
Dim = int | tuple[int, ...]

MAX_LINE_WIDTH = 200
PREC = 4
FLOATMODE = "maxprec_equal"
numpy.set_printoptions(precision=PREC, linewidth=MAX_LINE_WIDTH, floatmode=FLOATMODE)
cupy.set_printoptions(precision=PREC, linewidth=MAX_LINE_WIDTH, floatmode=FLOATMODE)


class Shape(tuple):
    def __repr__(self) -> str:
        return f"shape({super().__repr__().replace("(", "").replace(")", "")})"


ShapeLike: TypeAlias = Shape | tuple[int, ...]


def get_available_devices() -> list[str]:
    return ["cpu"] + [f"cuda:{i}" for i in range(cupy.cuda.runtime.getDeviceCount())]


def gpu_available():
    return cupy.cuda.is_available()


def set_random_seed(seed: int):
    numpy.random.seed(seed)
    if gpu_available():
        cupy.random.seed(seed)


class Device:
    def __init__(self, dev_type: str):
        dev_type, dev_id = _get_type_and_id(dev_type)
        self.dev_type = dev_type
        self.dev_id = dev_id
        self.xp = numpy if dev_type == "cpu" else cupy

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Device)
            and other.dev_type == self.dev_type
            and other.dev_id == self.dev_id
        )

    def __repr__(self) -> str:
        id_suffix = f":{self.dev_id}" if self.dev_type == "cuda" else ""
        return f"device('{self.dev_type}{id_suffix}')"

    def __enter__(self) -> None:
        if self.dev_type == "cpu":
            return None
        return cupy.cuda.Device(self.dev_id).__enter__()

    def __exit__(self, *args: Any) -> None:
        if self.dev_type == "cpu":
            return None
        return cupy.cuda.Device(self.dev_id).__exit__(*args)


DeviceLike: TypeAlias = Device | str


def _get_type_and_id(device_type: str) -> tuple[str, Optional[int]]:
    match = re.match(r"(?P<type>cpu|cuda)(?::(?P<id>\d+))?", device_type)
    if match:
        device_type = match.group("type")
        if device_type == "cuda":
            assert gpu_available()
        device_id = match.group("id")
        return (device_type, None if device_id is None else int(device_id))
    raise ValueError(f"Unknown device: {device_type}")


def get_array_device(x: ArrayLike) -> Device:
    return Device("cuda:0") if isinstance(x, cupy.ndarray) else Device("cpu")


def select_device(device: Optional[DeviceLike]) -> Device:
    if isinstance(device, Device):
        return device
    return Device(device or "cpu")


def parse_device(device: DeviceLike) -> Device:
    return device if isinstance(device, Device) else Device(device)


def move_to_device(data: ArrayLike, device: Device) -> ArrayLike:
    if device == Device("cpu"):
        return cupy.asnumpy(data)
    return cupy.asarray(data)


def array_to_string(data: ArrayLike, prefix: str) -> str:
    device = get_array_device(data)
    return device.xp.array2string(
        data,
        max_line_width=MAX_LINE_WIDTH,
        precision=PREC,
        separator=", ",
        prefix=prefix,
        floatmode=FLOATMODE,
    )
