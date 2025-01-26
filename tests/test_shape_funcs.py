"""Tests for shape functions"""

import pytest
import torch

import auto_compyute as ac

from .utils import close, get_data


def _shape_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_data(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad)


ac_x1, torch_x1 = get_data((10, 20))
ac_x2, torch_x2 = get_data((10, 20, 30))
xs = ((ac_x1, torch_x1), (ac_x2, torch_x2))
keys = (1, [1, 1, 2], ac.tensor([1, 2]))


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("key", keys)
def test_select(x, key):
    """Selection function test"""
    ac_x, torch_x = x
    ac_y = ac_x[key]
    torch_y = torch_x[torch.tensor(key.data) if isinstance(key, ac.Tensor) else key]
    _shape_function_verify(ac_x, torch_x, ac_y, torch_y)
