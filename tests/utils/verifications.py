"""Test result verification utils."""

import numpy as np
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats

DEFAULT_TOL = 1e-4


def allclose(
    ac_array: ac.backends.Array, torch_tensor: torch.Tensor, tol: float = DEFAULT_TOL
) -> bool:
    # torch.allclose does not support some scenarios
    return np.allclose(ac_array, torch_tensor.detach().numpy(), atol=tol, rtol=tol)


def verify_op(
    ac_xs: tuple[ac.Tensor | float, ...],
    ac_y: ac.Tensor,
    torch_xs: tuple[torch.Tensor | float, ...],
    torch_y: torch.Tensor,
    *,
    tol: float = DEFAULT_TOL,
) -> None:
    """Asserts closeness for the output and input gradient of a AutoCompyute and Torch function"""
    assert allclose(ac_y.as_numpy(), torch_y, tol=tol)
    if not ac_y.req_grad:
        return
    ac_y_grad, torch_y_grad = get_random_floats(ac_y.shape, False)
    ac_y.backward(ac_y_grad.data)
    torch_y.backward(torch_y_grad)
    for ac_x, torch_x in zip(ac_xs, torch_xs):
        if isinstance(ac_x, ac.Tensor) and ac_x.req_grad:
            assert allclose(ac_x.grad, torch_x.grad, tol=tol)
