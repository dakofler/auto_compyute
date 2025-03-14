"""
# AutoCompyute

Lightweight Autograd Engine in Pure Python

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/dakofler/auto_compyute)

AutoCompyute is a deep learning library that provides automatic differentiation using only [NumPy](https://numpy.org/) as the backend for computation ([CuPy](https://cupy.dev/) can be used as a drop-in replacement for NumPy). It is designed for simplicity and performance and enables you to train deep learning models with minimal dependencies while leveraging GPU acceleration. The package supports:

- Tensor operations with gradient tracking.
- Neural network layers and loss functions.
- Performant computation for both CPU and GPU.
- A focus on clarity and simplicity.
"""

import importlib.metadata

from . import nn
from .autograd import *
from .backends import *
from .dtypes import *
from .tensor_factory import *
from .tensor_functions import *

__version__ = importlib.metadata.version("auto_compyute")
