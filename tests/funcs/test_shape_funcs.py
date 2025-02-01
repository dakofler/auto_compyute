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


def _combine_function_verify(x1, torch_x1, x2, torch_x2, x3, torch_x3, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x1.grad, torch_x1.grad)
    assert close(x2.grad, torch_x2.grad)
    assert close(x3.grad, torch_x3.grad)


comb_ac_x1, comb_torch_x1 = get_random_floats((10, 5))
comb_ac_x2, comb_torch_x2 = get_random_floats((10, 5))
comb_ac_x3, comb_torch_x3 = get_random_floats((10, 5))
comb_x1s = ((comb_ac_x1, comb_torch_x1),)
comb_x2s = ((comb_ac_x2, comb_torch_x2),)
comb_x3s = ((comb_ac_x3, comb_torch_x3),)
comb_dims = (0, 1)


@pytest.mark.parametrize("x1", comb_x1s)
@pytest.mark.parametrize("x2", comb_x2s)
@pytest.mark.parametrize("x3", comb_x3s)
@pytest.mark.parametrize("dim", comb_dims)
def test_stack(x1, x2, x3, dim):
    """Stack function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_x3, torch_x3 = x3
    ac_y = ac.stack(ac_x1, ac_x2, ac_x3, dim=dim)
    torch_y = torch.stack((torch_x1, torch_x2, torch_x3), dim=dim)
    _combine_function_verify(
        ac_x1, torch_x1, ac_x2, torch_x2, ac_x3, torch_x3, ac_y, torch_y
    )


@pytest.mark.parametrize("x1", comb_x1s)
@pytest.mark.parametrize("x2", comb_x2s)
@pytest.mark.parametrize("x3", comb_x3s)
@pytest.mark.parametrize("dim", comb_dims)
def test_concat(x1, x2, x3, dim):
    """Concat function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_x3, torch_x3 = x3
    ac_y = ac.concat(ac_x1, ac_x2, ac_x3, dim=dim)
    torch_y = torch.concat((torch_x1, torch_x2, torch_x3), dim=dim)
    _combine_function_verify(
        ac_x1, torch_x1, ac_x2, torch_x2, ac_x3, torch_x3, ac_y, torch_y
    )
