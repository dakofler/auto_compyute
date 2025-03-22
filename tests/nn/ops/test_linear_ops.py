"""Tests for linear operations."""

import pytest
import torch.nn.functional as tF

import auto_compyute.nn.functional as F
from tests.utils import close, get_random_floats


def _linear_function_verify(x, torch_x, w, torch_w, b, torch_b, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad)
    assert close(w.grad, torch_w.grad, tol=1e-4)
    if b is not None:
        assert close(b.grad, torch_b.grad)


ac_x1, torch_x1 = get_random_floats((16, 32))
ac_x2, torch_x2 = get_random_floats((8, 16, 32))
ac_w1, torch_w1 = get_random_floats((64, 32))
ac_b1, torch_b1 = get_random_floats((64,))
xs = ((ac_x1, torch_x1), (ac_x2, torch_x2))
ws = ((ac_w1, torch_w1),)
bs = ((ac_b1, torch_b1), (None, None))


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("w", ws)
@pytest.mark.parametrize("b", bs)
def test_linear(x, w, b):
    """Linear function test"""
    ac_x, torch_x = x
    ac_w, torch_w = w
    ac_b, torch_b = b
    ac_y = F.linear(ac_x, ac_w, ac_b)
    torch_y = tF.linear(torch_x, torch_w, torch_b)
    _linear_function_verify(ac_x, torch_x, ac_w, torch_w, ac_b, torch_b, ac_y, torch_y)
