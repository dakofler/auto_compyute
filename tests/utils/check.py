"""Test check utils."""

import numpy as np
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats


def close(ac_array: ac.backends.Array, torch_tensor: torch.Tensor, tol: float = 1e-4) -> bool:
    return np.allclose(ac_array, torch_tensor.detach().numpy(), atol=tol, rtol=tol)


def single_input_op_check(
    ac_x: ac.Tensor, ac_y: ac.Tensor, torch_x: torch.Tensor, torch_y: torch.Tensor
) -> None:
    assert close(ac_y.numpy(), torch_y)
    if not ac_y.req_grad:
        return
    ac_y_grad, torch_y_grad = get_random_floats(ac_y.shape, False)
    ac_y.backward(ac_y_grad.data)
    torch_y.backward(torch_y_grad)
    assert close(ac_x.grad, torch_x.grad)


def dual_input_op_check(
    ac_x_1: ac.Tensor,
    ac_x_2: ac.Tensor,
    ac_y: ac.Tensor,
    torch_x_1: torch.Tensor,
    torch_x_2: torch.Tensor,
    torch_y: torch.Tensor,
) -> None:
    assert close(ac_y.numpy(), torch_y)
    if not ac_y.req_grad:
        return
    ac_y_grad, torch_y_grad = get_random_floats(ac_y.shape, False)
    ac_y.backward(ac_y_grad.data)
    torch_y.backward(torch_y_grad)
    if isinstance(ac_x_1, ac.Tensor):
        assert close(ac_x_1.grad, torch_x_1.grad)
    if isinstance(ac_x_2, ac.Tensor):
        assert close(ac_x_2.grad, torch_x_2.grad)


def triple_input_op_check(
    ac_x_1: ac.Tensor,
    ac_x_2: ac.Tensor,
    ac_x_3: ac.Tensor,
    ac_y: ac.Tensor,
    torch_x_1: torch.Tensor,
    torch_x_2: torch.Tensor,
    torch_x_3: torch.Tensor,
    torch_y: torch.Tensor,
) -> None:
    assert close(ac_y.numpy(), torch_y)
    if not ac_y.req_grad:
        return
    ac_y_grad, torch_y_grad = get_random_floats(ac_y.shape, False)
    ac_y.backward(ac_y_grad.data)
    torch_y.backward(torch_y_grad)
    if isinstance(ac_x_1, ac.Tensor):
        assert close(ac_x_1.grad, torch_x_1.grad)
    if isinstance(ac_x_2, ac.Tensor):
        assert close(ac_x_2.grad, torch_x_2.grad)
    if isinstance(ac_x_3, ac.Tensor):
        assert close(ac_x_3.grad, torch_x_3.grad)
