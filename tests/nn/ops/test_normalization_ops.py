"""Tests for normalization operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_ones, get_random_floats, get_zeros
from tests.utils.verifications import verify_op

NORM_DIM = 32
W_RANDOM_FLOAT_TENSORS = (get_random_floats((NORM_DIM,)),)
B_RANDOM_FLOAT_TENSORS = (get_random_floats((NORM_DIM,)),)
BN_IN_SHAPES = ((16, NORM_DIM), (16, NORM_DIM, 8), (16, NORM_DIM, 28, 28))
BN_X_RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in BN_IN_SHAPES)
BN_RMEAN_ZEROS_TENSORS = (get_zeros((NORM_DIM,), req_grad=False),)
BN_RVAR_ONES_TENSORS = (get_ones((NORM_DIM,), req_grad=False),)
BN_MS = (0.1, 0.2)
LN_IN_SHAPES = ((16, NORM_DIM), (16, 8, NORM_DIM), (16, 8, 8, NORM_DIM))
LN_X_RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in LN_IN_SHAPES)


@pytest.mark.parametrize("x", BN_X_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("rmean", BN_RMEAN_ZEROS_TENSORS)
@pytest.mark.parametrize("rvar", BN_RVAR_ONES_TENSORS)
@pytest.mark.parametrize("w", W_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("b", B_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("m", BN_MS)
def test_batchnorm(
    x: tuple[ac.Tensor, torch.Tensor],
    rmean: tuple[ac.Tensor, torch.Tensor],
    rvar: tuple[ac.Tensor, torch.Tensor],
    w: tuple[ac.Tensor, torch.Tensor],
    b: tuple[ac.Tensor, torch.Tensor],
    m: float,
) -> None:
    ac_x, torch_x = tuple(zip(*[x, rmean, rvar, w, b]))
    ac_y = ac.nn.functional.batchnorm(*ac_x, momentum=m, eps=1e-5, training=True)
    torch_y = torch.nn.functional.batch_norm(*torch_x, training=True, momentum=m, eps=1e-5)
    verify_op(ac_x, ac_y, torch_x, torch_y)


@pytest.mark.parametrize("x", LN_X_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("w", W_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("b", B_RANDOM_FLOAT_TENSORS)
def test_layernorm(
    x: tuple[ac.Tensor, torch.Tensor],
    w: tuple[ac.Tensor, torch.Tensor],
    b: tuple[ac.Tensor, torch.Tensor],
) -> None:
    ac_x, torch_x = tuple(zip(*[x, w, b]))
    ac_y = ac.nn.functional.layernorm(*ac_x, eps=1e-5)
    torch_x_, torch_w, torch_b = torch_x
    torch_y = torch.nn.functional.layer_norm(
        torch_x_, (torch_x_.shape[-1],), torch_w, torch_b, eps=1e-5
    )
    verify_op(ac_x, ac_y, torch_x, torch_y)
