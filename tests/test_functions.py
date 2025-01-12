"""Tests for functions"""

import pytest

from .utils import close, get_data


def binary_function_head(shape1, shape2):
    x, torch_x = get_data(shape1)
    y, torch_y = get_data(shape2)
    return x, y, torch_x, torch_y


def binary_function_tail(x, y, z, torch_x, torch_y, torch_z):
    assert close(z.data, torch_z)
    dy, torch_dy = get_data(z.shape)
    z.backward(dy.data)
    torch_z.backward(torch_dy)
    assert close(x.grad, torch_x.grad)
    assert close(y.grad, torch_y.grad)


shape1 = ((8, 16, 32),)
shape2 = ((32,), (16, 32), (8, 16, 32))


@pytest.mark.parametrize("shape1", shape1)
@pytest.mark.parametrize("shape2", shape2)
def test_add(shape1, shape2):
    x, y, torch_x, torch_y = binary_function_head(shape1, shape2)
    z = x + y
    torch_z = torch_x + torch_y
    binary_function_tail(x, y, z, torch_x, torch_y, torch_z)


@pytest.mark.parametrize("shape1", shape1)
@pytest.mark.parametrize("shape2", shape2)
def test_sub(shape1, shape2):
    x, y, torch_x, torch_y = binary_function_head(shape1, shape2)
    z = x - y
    torch_z = torch_x - torch_y
    binary_function_tail(x, y, z, torch_x, torch_y, torch_z)


@pytest.mark.parametrize("shape1", shape1)
@pytest.mark.parametrize("shape2", shape2)
def test_mul(shape1, shape2):
    x, y, torch_x, torch_y = binary_function_head(shape1, shape2)
    z = x * y
    torch_z = torch_x * torch_y
    binary_function_tail(x, y, z, torch_x, torch_y, torch_z)


@pytest.mark.parametrize("shape1", shape1)
@pytest.mark.parametrize("shape2", shape2)
def test_div(shape1, shape2):
    x, y, torch_x, torch_y = binary_function_head(shape1, shape2)
    z = x / y
    torch_z = torch_x / torch_y
    binary_function_tail(x, y, z, torch_x, torch_y, torch_z)
