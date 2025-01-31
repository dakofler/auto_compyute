"""Tests for shape functions"""

import pytest
import torch

import auto_compyute as ac

from ..utils import close, get_random_floats


def _shape_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad)


ac_x1, torch_x1 = get_random_floats((10, 20))
ac_x2, torch_x2 = get_random_floats((10, 20, 30))
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


transpose_dims = ((0, 1), (-1, -2))


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dims", transpose_dims)
def test_transpose(x, dims):
    """Transpose function test"""
    ac_x, torch_x = x
    ac_y = ac_x.transpose(*dims)
    torch_y = torch_x.transpose(*dims)
    _shape_function_verify(ac_x, torch_x, ac_y, torch_y)


view_xs = ((ac_x1, torch_x1),)
view_shapes = ((20, 10), (10, 10, 2))


@pytest.mark.parametrize("x", view_xs)
@pytest.mark.parametrize("shape", view_shapes)
def test_view(x, shape):
    """View function test"""
    ac_x, torch_x = x
    ac_y = ac_x.view(shape)
    torch_y = torch_x.view(*shape)
    _shape_function_verify(ac_x, torch_x, ac_y, torch_y)


split_sizes = (2, 5)
split_dims = (0, 1, -1)


@pytest.mark.parametrize("x", view_xs)
@pytest.mark.parametrize("split_size", split_sizes)
@pytest.mark.parametrize("dim", split_dims)
def test_split(x, split_size, dim):
    """Split function test"""
    ac_x, torch_x = x
    ac_ys = ac_x.split(split_size, dim)
    torch_ys = torch_x.split(split_size, dim)
    for ac_y, torch_y in zip(ac_ys, torch_ys):
        _shape_function_verify(ac_x, torch_x, ac_y, torch_y)
