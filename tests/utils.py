"""Test utils"""

import numpy as np
import torch
from torch import Tensor

import auto_compyute as ac
from auto_compyute.devices import Array, Shape

np.random.seed(42)


def close(ac_in: Array, torch_in: Tensor, tol: float = 1e-5):
    """Checks wheter an array and a PyTorch tensor are close."""
    return np.allclose(ac_in, torch_in.detach().numpy(), atol=tol, rtol=tol)


def get_data(shape: Shape, requires_grad: bool = True):
    """Returns randomly initialized data as tensors"""
    x = ac.randn(shape, requires_grad=requires_grad)
    torch_x = torch.tensor(x.data, requires_grad=requires_grad)
    return x, torch_x
