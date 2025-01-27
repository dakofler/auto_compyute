"""Backend devices"""

from types import ModuleType
from typing import Optional, TypeAlias

import cupy
import numpy

__all__ = ["cpu", "cuda"]

numpy.set_printoptions(precision=4, linewidth=80, floatmode="maxprec_equal")
cupy.set_printoptions(precision=4, linewidth=80, floatmode="maxprec_equal")

Array: TypeAlias = cupy.ndarray | numpy.ndarray
Scalar: TypeAlias = int | float
Shape = tuple[int, ...]
Dim = int | tuple[int, ...]


class Device:
    m: ModuleType = numpy

    def __repr__(self):
        return f"Device({self.__class__.__name__})"


class CPU(Device):
    m = numpy


class CUDA(Device):
    m = cupy


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
