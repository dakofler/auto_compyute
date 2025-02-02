"""Backend devices"""

from types import ModuleType
from typing import Optional, TypeAlias

import cupy
import numpy

__all__ = ["cpu", "cuda"]

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
    backend: ModuleType = numpy

    def __repr__(self):
        return f"Device({self.__class__.__name__})"


class CPU(Device):
    backend = numpy


class CUDA(Device):
    backend = cupy


cpu = CPU()
cuda = CUDA()


def get_array_device(x: Array) -> Device:
    return cuda if isinstance(x, cupy.ndarray) else cpu


def select_device(device: Optional[Device]) -> Device:
    return device if device is not None else cpu


def move_to_device(data: Array, device: Device) -> Array:
    if device == cpu:
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
