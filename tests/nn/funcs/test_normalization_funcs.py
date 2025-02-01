"""Tests for convolution functions"""

import pytest
import torch.nn.functional as tF

import auto_compyute.nn.functional as F

from ...utils import close, get_ones, get_random_floats, get_zeros


def _batchnorm_function_verify(x, torch_x, w, torch_w, b, torch_b, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad)
    assert close(w.grad, torch_w.grad, tol=1e-4)
    assert close(b.grad, torch_b.grad, tol=1e-4)


ac_x1, torch_x1 = get_random_floats((16, 32))
ac_x2, torch_x2 = get_random_floats((16, 32, 8))
ac_x3, torch_x3 = get_random_floats((16, 32, 28, 28))
ac_w1, torch_w1 = get_random_floats((32,))
ac_b1, torch_b1 = get_random_floats((32,))
xs = ((ac_x1, torch_x1), (ac_x2, torch_x2), (ac_x3, torch_x3))
ws = ((ac_w1, torch_w1),)
bs = ((ac_b1, torch_b1),)
ms = (0.1, 0.2)
trainings = (True, False)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("w", ws)
@pytest.mark.parametrize("b", bs)
@pytest.mark.parametrize("m", ms)
@pytest.mark.parametrize("training", trainings)
def test_batchnorm(x, w, b, m, training):
    """Batchnorm function test"""
    ac_x, torch_x = x
    ac_w, torch_w = w
    ac_b, torch_b = b
    ac_rmean, torch_rmean = get_zeros((ac_x.shape[1],))
    ac_rvar, torch_rvar = get_ones((ac_x.shape[1],))
    ac_y, *_ = F.batchnorm(ac_x, ac_rmean, ac_rvar, ac_w, ac_b, m, 1e-5, training)
    torch_y = tF.batch_norm(
        torch_x, torch_rmean, torch_rvar, torch_w, torch_b, training, m, 1e-5
    )
    _batchnorm_function_verify(
        ac_x, torch_x, ac_w, torch_w, ac_b, torch_b, ac_y, torch_y
    )
