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


def select_dtype(dtype: Optional[DType]) -> DType:
    return dtype or float32


def is_float(dtype: DType) -> bool:
    return any(dtype == d for d in [float16, float32, float64])


def is_int(dtype: DType) -> bool:
    return any(dtype == d for d in [int16, int32, int64])
