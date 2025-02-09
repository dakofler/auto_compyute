"""
# AutoCompyute
## Lightweight Deep Learning with Pure Python

AutoCompyute is a lightweight and efficient deep learning library that provides automatic
differentiation using only NumPy and CuPy as backends. Designed for simplicity and performance, it
enables users to build and train deep learning models with minimal dependencies while leveraging
GPU acceleration via CuPy. The package supports:

- Flexible tensor operations with gradient tracking.
- Customizable neural network layers and loss functions.
- Optimized computation for both CPU and GPU.
- A focus on clarity, making it ideal for research and education.

Whether you're exploring the fundamentals of autograd or developing deep learning models, this
library offers a pure Python solution with a streamlined API.
"""

import pathlib

from . import nn
from .array_factory import *
from .array_functions import *
from .autograd import *
from .backends import *
from .dtypes import *

__version__ = (pathlib.Path(__file__).parents[0] / "VERSION").read_text(
    encoding="utf-8"
)
