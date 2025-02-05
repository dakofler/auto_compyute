"""Test utils"""

import numpy as np
import torch
from torch import Tensor

import auto_compyute as ac
from auto_compyute.backends import ArrayLike, ShapeLike

np.random.seed(0)


def close(ac_in: ArrayLike, torch_in: Tensor, tol: float = 1e-5):
    return np.allclose(ac_in, torch_in.detach().numpy(), atol=tol, rtol=tol)


def get_random_floats(shape: ShapeLike, req_grad: bool = True):
    x = ac.randn(*shape, req_grad=req_grad)
    torch_x = torch.tensor(x.data, requires_grad=req_grad)
    return x, torch_x


def get_random_positive_floats(shape: ShapeLike, req_grad: bool = True):
    x = ac.randn(*shape).abs()
    x.req_grad = req_grad
    torch_x = torch.tensor(x.data, requires_grad=req_grad)
    return x, torch_x


def get_random_ints(shape: ShapeLike, low: int, high: int):
    x = ac.randi(*shape, low=low, high=high, dtype=ac.int64)
    torch_x = torch.tensor(x.data)
    return x, torch_x


def get_random_bools(shape: ShapeLike):
    x = ac.randn(*shape) < 0
    torch_x = torch.tensor(x.data)
    return x, torch_x


def get_ones(shape: ShapeLike, req_grad: bool = False):
    x = ac.ones(*shape, req_grad=req_grad)
    torch_x = torch.tensor(x.data, requires_grad=req_grad)
    return x, torch_x


def get_zeros(shape: ShapeLike, req_grad: bool = False):
    x = ac.zeros(*shape, req_grad=req_grad)
    torch_x = torch.tensor(x.data, requires_grad=req_grad)
    return x, torch_x
