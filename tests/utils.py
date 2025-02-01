"""Test utils"""

import numpy as np
import torch
from torch import Tensor

import auto_compyute as ac
from auto_compyute.backends import Array, Shape

np.random.seed(0)


def close(ac_in: Array, torch_in: Tensor, tol: float = 1e-5):
    return np.allclose(ac_in, torch_in.detach().numpy(), atol=tol, rtol=tol)


def get_random_floats(shape: Shape, requires_grad: bool = True):
    x = ac.randn(shape, requires_grad=requires_grad)
    torch_x = torch.tensor(x.data, requires_grad=requires_grad)
    return x, torch_x


def get_random_positive_floats(shape: Shape, requires_grad: bool = True):
    x = ac.randn(shape).abs()
    x.requires_grad = requires_grad
    torch_x = torch.tensor(x.data, requires_grad=requires_grad)
    return x, torch_x


def get_random_ints(shape: Shape, low: int, high: int):
    x = ac.randi(shape, low, high, dtype=ac.int64)
    torch_x = torch.tensor(x.data)
    return x, torch_x


def get_ones(shape: Shape, requires_grad: bool = False):
    x = ac.ones(shape, requires_grad=requires_grad)
    torch_x = torch.tensor(x.data, requires_grad=requires_grad)
    return x, torch_x


def get_zeros(shape: Shape, requires_grad: bool = False):
    x = ac.zeros(shape, requires_grad=requires_grad)
    torch_x = torch.tensor(x.data, requires_grad=requires_grad)
    return x, torch_x
