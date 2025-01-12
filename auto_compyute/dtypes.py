"""Array data types"""

from typing import Optional, TypeAlias

import numpy as np

__all__ = [
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool_",
    "complex64",
    "complex128",
]

DType: TypeAlias = type


# Float types
float16 = np.float16
float32 = np.float32
float64 = np.float64

# Integer types
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

# Unsigned integer types
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

# Boolean type
bool_ = np.bool_

# Complex types
complex64 = np.complex64
complex128 = np.complex128


def select_dtype(device: Optional[DType]) -> DType:
    return device if device is not None else float32
