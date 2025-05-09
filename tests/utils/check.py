"""Test check utils."""

import numpy as np
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats


def close(ac_array: ac.backends.Array, torch_tensor: torch.Tensor, tol: float = 1e-5) -> bool:
    return np.allclose(ac_array, torch_tensor.detach().numpy(), atol=tol, rtol=tol)


def single_input_op_check(
    ac_in_tensor: ac.Tensor,
    ac_out_tensor: ac.Tensor,
    torch_in_tensor: torch.Tensor,
    torch_out_tensor: torch.Tensor,
) -> None:
    # check out
    assert close(ac_out_tensor.numpy(), torch_out_tensor)

    # compute gradients
    ac_out_tensor_grad, torch_out_tensor_grad = get_random_floats(
        ac_out_tensor.shape, req_grad=False
    )
    ac_out_tensor.backward(ac_out_tensor_grad.data)
    torch_out_tensor.backward(torch_out_tensor_grad)

    # check gradients
    assert close(ac_in_tensor.grad, torch_in_tensor.grad)


def dual_input_op_check(
    ac_in_tensor1: ac.Tensor,
    ac_in_tensor2: ac.Tensor,
    ac_out_tensor: ac.Tensor,
    torch_in_tensor1: torch.Tensor,
    torch_in_tensor2: torch.Tensor,
    torch_out_tensor: torch.Tensor,
) -> None:
    # check out
    assert close(ac_out_tensor.numpy(), torch_out_tensor)

    # compute gradients
    ac_out_tensor_grad, torch_out_tensor_grad = get_random_floats(
        ac_out_tensor.shape, req_grad=False
    )
    ac_out_tensor.backward(ac_out_tensor_grad.data)
    torch_out_tensor.backward(torch_out_tensor_grad)

    # check gradients
    assert close(ac_in_tensor1.grad, torch_in_tensor1.grad)
    if isinstance(ac_in_tensor2, ac.Tensor):
        assert close(ac_in_tensor2.grad, torch_in_tensor2.grad)
