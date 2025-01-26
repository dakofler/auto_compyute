"""Tests for reduce functions"""

import pytest

from .utils import close, get_data


def _reduce_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_data(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad)


ac_x1, torch_x1 = get_data((10, 20))
ac_x2, torch_x2 = get_data((10, 20, 30))
xs = ((ac_x1, torch_x1), (ac_x2, torch_x2))
dims = (None, 0, (0, 1))
keepdimses = (False, True)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("keepdims", keepdimses)
def test_sum(x, dim, keepdims):
    """Sum function test"""
    ac_x, torch_x = x
    ac_y = ac_x.sum(dim, keepdims=keepdims)
    torch_y = torch_x.sum(dim, keepdims=keepdims)
    _reduce_function_verify(ac_x, torch_x, ac_y, torch_y)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("keepdims", keepdimses)
def test_mean(x, dim, keepdims):
    """Mean function test"""
    ac_x, torch_x = x
    ac_y = ac_x.mean(dim, keepdims=keepdims)
    torch_y = torch_x.mean(dim, keepdims=keepdims)
    _reduce_function_verify(ac_x, torch_x, ac_y, torch_y)


ddofs = (0, 1)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("ddof", ddofs)
@pytest.mark.parametrize("keepdims", keepdimses)
def test_var(x, dim, ddof, keepdims):
    """Var function test"""
    ac_x, torch_x = x
    ac_y = ac_x.var(dim, ddof=ddof, keepdims=keepdims)
    torch_y = torch_x.var(dim, correction=ddof, keepdim=keepdims)
    _reduce_function_verify(ac_x, torch_x, ac_y, torch_y)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("keepdims", keepdimses)
def test_std(x, dim, keepdims):
    """Std function test"""
    ac_x, torch_x = x
    ac_y = ac_x.std(dim, keepdims=keepdims)
    torch_y = torch_x.std(dim, keepdims=keepdims)
    _reduce_function_verify(ac_x, torch_x, ac_y, torch_y)


min_max_dims = (None, 0, 1)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dim", min_max_dims)
@pytest.mark.parametrize("keepdims", keepdimses)
def test_max(x, dim, keepdims):
    """Max function test"""
    ac_x, torch_x = x
    ac_y = ac_x.max(dim, keepdims=keepdims)
    if dim is None and not keepdims:
        torch_y = torch_x.max()
    elif dim is None:
        return
    else:
        torch_y = torch_x.max(dim, keepdims=keepdims)[0]
    _reduce_function_verify(ac_x, torch_x, ac_y, torch_y)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dim", min_max_dims)
@pytest.mark.parametrize("keepdims", keepdimses)
def test_min(x, dim, keepdims):
    """Min function test"""
    ac_x, torch_x = x
    ac_y = ac_x.min(dim, keepdims=keepdims)
    if dim is None and not keepdims:
        torch_y = torch_x.min()
    elif dim is None:
        return
    else:
        torch_y = torch_x.min(dim, keepdims=keepdims)[0]
    _reduce_function_verify(ac_x, torch_x, ac_y, torch_y)
