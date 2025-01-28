"""Tests for activation functions"""

import pytest
import torch.nn.functional as tF

import auto_compyute.nn.functional as F

from ...utils import close, get_data


def _unary_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_data(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad)


ac_x1, torch_x1 = get_data((10, 20))
xs = ((ac_x1, torch_x1),)
dims = (0, 1)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dim", dims)
def test_softmax(x, dim):
    """Softmax function test"""
    ac_x, torch_x = x
    ac_y = F.softmax(ac_x, dim=dim)
    torch_y = tF.softmax(torch_x, dim)
    _unary_function_verify(ac_x, torch_x, ac_y, torch_y)
