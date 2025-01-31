"""Tests for unary functions"""

import pytest
import torch

from ..utils import close, get_random_floats


def _unary_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad)


ac_x1, torch_x1 = get_random_floats((10, 20))
ac_x2, torch_x2 = get_random_floats((10, 20, 30))
xs = ((ac_x1, torch_x1), (ac_x2, torch_x2))


@pytest.mark.parametrize("x", xs)
def test_exp(x):
    """Exp function test"""
    ac_x, torch_x = x
    ac_y = ac_x.exp()
    torch_y = torch.exp(torch_x)
    _unary_function_verify(ac_x, torch_x, ac_y, torch_y)


@pytest.mark.parametrize("x", xs)
def test_tanh(x):
    """Tanh function test"""
    ac_x, torch_x = x
    ac_y = ac_x.tanh()
    torch_y = torch.tanh(torch_x)
    _unary_function_verify(ac_x, torch_x, ac_y, torch_y)
