"""Test tensor initialization utils."""

import torch

import auto_compyute as ac
from auto_compyute.backends import ShapeLike

ac.set_random_seed(0)


def get_random_floats(shape: ShapeLike, req_grad: bool = True) -> tuple[ac.Tensor, torch.Tensor]:
    ac_tensor = ac.randn(*shape, req_grad=req_grad)
    torch_tensor = torch.tensor(ac_tensor.as_numpy(), requires_grad=req_grad)
    return ac_tensor, torch_tensor


def get_random_positive_floats(
    shape: ShapeLike, req_grad: bool = True
) -> tuple[ac.Tensor, torch.Tensor]:
    ac_tensor = ac.randn(*shape).abs()
    ac_tensor.req_grad = req_grad
    torch_tensor = torch.tensor(ac_tensor.as_numpy(), requires_grad=req_grad)
    return ac_tensor, torch_tensor


def get_random_ints(shape: ShapeLike, low: int, high: int) -> tuple[ac.Tensor, torch.Tensor]:
    ac_tensor = ac.randi(*shape, low=low, high=high, dtype=ac.int64)
    torch_tensor = torch.tensor(ac_tensor.as_numpy())
    return ac_tensor, torch_tensor


def get_random_bools(shape: ShapeLike) -> tuple[ac.Tensor, torch.Tensor]:
    ac_tensor = ac.randn(*shape) < 0
    torch_tensor = torch.tensor(ac_tensor.as_numpy())
    return ac_tensor, torch_tensor


def get_ones(shape: ShapeLike, req_grad: bool = False) -> tuple[ac.Tensor, torch.Tensor]:
    ac_tensor = ac.ones(*shape, req_grad=req_grad)
    torch_tensor = torch.ones(shape, requires_grad=req_grad)
    return ac_tensor, torch_tensor


def get_zeros(shape: ShapeLike, req_grad: bool = False) -> tuple[ac.Tensor, torch.Tensor]:
    ac_tensor = ac.zeros(*shape, req_grad=req_grad)
    torch_tensor = torch.zeros(shape, requires_grad=req_grad)
    return ac_tensor, torch_tensor
