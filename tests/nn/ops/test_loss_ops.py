"""Tests for loss function operations."""

import pytest
import torch.nn.functional as tF

import auto_compyute.nn.functional as F

from ...utils import close, get_random_floats, get_random_ints


def _loss_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    y.backward()
    torch_y.backward()
    assert close(x.grad, torch_x.grad)


ac_x1, torch_x1 = get_random_floats((4, 8))
ac_x2, torch_x2 = get_random_floats((4, 8, 16))
ac_x3, torch_x3 = get_random_floats((4, 8, 16, 32))
xs = ((ac_x1, torch_x1), (ac_x2, torch_x2), (ac_x3, torch_x3))
reductions = ("mean", "sum")

mse_ac_t1, mse_torch_t1 = get_random_floats((4, 8))
mse_ac_t2, mse_torch_t2 = get_random_floats((4, 8, 16))
mse_ac_t3, mse_torch_t3 = get_random_floats((4, 8, 16, 32))
mse_ts = (
    (mse_ac_t1, mse_torch_t1),
    (mse_ac_t2, mse_torch_t2),
    (mse_ac_t3, mse_torch_t3),
)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("t", mse_ts)
@pytest.mark.parametrize("reduction", reductions)
def test_mse_loss(x, t, reduction):
    """MSE function test"""
    ac_x, torch_x = x
    ac_t, torch_t = t

    # skip tests with shape mismatch
    if ac_x.shape != ac_t.shape:
        return

    ac_y = F.mse_loss(ac_x, ac_t, reduction)
    torch_y = tF.mse_loss(torch_x, torch_t, reduction=reduction)
    _loss_function_verify(ac_x, torch_x, ac_y, torch_y)


ce_ac_t1, ce_torch_t1 = get_random_ints((4,), 0, 8)
ce_ac_t2, ce_torch_t2 = get_random_ints((4, 8), 0, 16)
ce_ac_t3, ce_torch_t3 = get_random_ints((4, 8, 16), 0, 32)
ce_ts = (
    (ce_ac_t1, ce_torch_t1),
    (ce_ac_t2, ce_torch_t2),
    (ce_ac_t3, ce_torch_t3),
)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("t", ce_ts)
@pytest.mark.parametrize("reduction", reductions)
def test_cross_entropy_loss(x, t, reduction):
    """Cross entropy function test"""
    ac_x, torch_x = x
    ac_t, torch_t = t

    # skip tests with shape mismatch
    if ac_x.shape[:-1] != ac_t.shape:
        return

    ac_y = F.cross_entropy_loss(ac_x, ac_t, reduction=reduction)
    permutation = (0, ac_x.ndim - 1) + tuple(d for d in range(ac_x.ndim - 1) if d > 0)
    torch_y = tF.cross_entropy(
        torch_x.permute(*permutation), torch_t, reduction=reduction
    )
    _loss_function_verify(ac_x, torch_x, ac_y, torch_y)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("t", mse_ts)
@pytest.mark.parametrize("reduction", reductions)
def test_bce_loss(x, t, reduction):
    """BCE function test"""
    ac_x, torch_x = x
    ac_t, torch_t = t

    # skip tests with shape mismatch
    if ac_x.shape != ac_t.shape:
        return

    ac_y = F.bce_loss(ac_x, ac_t, reduction)
    torch_y = tF.binary_cross_entropy_with_logits(torch_x, torch_t, reduction=reduction)
    _loss_function_verify(ac_x, torch_x, ac_y, torch_y)
