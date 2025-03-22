"""Tests for binary operations."""

import pytest
import torch

from tests.utils import close, get_random_floats


def _binary_function_verify(x1, torch_x1, x2, torch_x2, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x1.grad, torch_x1.grad)
    if not isinstance(x2, float):
        assert close(x2.grad, torch_x2.grad)


ac_x11, torch_x11 = get_random_floats((10, 20))
x1s = ((ac_x11, torch_x11),)

ac_x21, torch_x21 = get_random_floats((10, 20))
ac_x22, torch_x22 = get_random_floats((20,))
ac_x23, torch_x23 = 2.0, 2.0
x2s = ((ac_x21, torch_x21), (ac_x22, torch_x22), (ac_x23, torch_x23))


@pytest.mark.parametrize("x1", x1s)
@pytest.mark.parametrize("x2", x2s)
def test_add(x1, x2):
    """Addition function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_y = ac_x1 + ac_x2
    torch_y = torch_x1 + torch_x2
    _binary_function_verify(ac_x1, torch_x1, ac_x2, torch_x2, ac_y, torch_y)


@pytest.mark.parametrize("x1", x1s)
@pytest.mark.parametrize("x2", x2s)
def test_sub(x1, x2):
    """Subtraction function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_y = ac_x1 - ac_x2
    torch_y = torch_x1 - torch_x2
    _binary_function_verify(ac_x1, torch_x1, ac_x2, torch_x2, ac_y, torch_y)


@pytest.mark.parametrize("x1", x1s)
@pytest.mark.parametrize("x2", x2s)
def test_mul(x1, x2):
    """Multiplication function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_y = ac_x1 * ac_x2
    torch_y = torch_x1 * torch_x2
    _binary_function_verify(ac_x1, torch_x1, ac_x2, torch_x2, ac_y, torch_y)


@pytest.mark.parametrize("x1", x1s)
@pytest.mark.parametrize("x2", x2s)
def test_truediv(x1, x2):
    """Division function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_y = ac_x1 / ac_x2
    torch_y = torch_x1 / torch_x2
    _binary_function_verify(ac_x1, torch_x1, ac_x2, torch_x2, ac_y, torch_y)


mm_ac_x11, mm_torch_x11 = get_random_floats((10, 20))
mm_ac_x12, mm_torch_x12 = get_random_floats((5, 10, 20))
mm_x1s = ((mm_ac_x11, mm_torch_x11), (mm_ac_x12, mm_torch_x12))

pow_ac_x21, pow_torch_x22 = get_random_floats((20, 3))
mm_x2s = ((pow_ac_x21, pow_torch_x22),)


@pytest.mark.parametrize("x1", mm_x1s)
@pytest.mark.parametrize("x2", mm_x2s)
def test_matmul(x1, x2):
    """Matrix multiplication function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_y = ac_x1 @ ac_x2
    torch_y = torch_x1 @ torch_x2
    _binary_function_verify(ac_x1, torch_x1, ac_x2, torch_x2, ac_y, torch_y)


pow_ac_x11, pow_torch_x11 = get_random_floats((20,))
pow_ac_x12, pow_torch_x12 = get_random_floats((10, 20))
pow_x1s = (
    (pow_ac_x11, pow_torch_x11),
    (pow_ac_x12, pow_torch_x12),
)

pow_ac_x21, pow_torch_x21 = 2.0, 2.0
pow_x2s = ((pow_ac_x21, pow_torch_x21),)


@pytest.mark.parametrize("x1", pow_x1s)
@pytest.mark.parametrize("x2", pow_x2s)
def test_pow(x1, x2):
    """Power function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_y = ac_x1**ac_x2
    torch_y = torch_x1**torch_x2
    _binary_function_verify(ac_x1, torch_x1, ac_x2, torch_x2, ac_y, torch_y)


min_ac_x11, min_torch_x11 = get_random_floats((20,))
min_ac_x12, min_torch_x12 = get_random_floats((10, 20))
min_x1s = (
    (min_ac_x11, min_torch_x11),
    (min_ac_x12, min_torch_x12),
)

min_ac_x21, min_torch_x21 = get_random_floats((20,))
min_ac_x22, min_torch_x22 = get_random_floats((10, 20))
min_ac_x23, min_torch_x23 = 0.0, torch.tensor(0.0)
min_x2s = (
    (min_ac_x21, min_torch_x21),
    (min_ac_x22, min_torch_x22),
    (min_ac_x23, min_torch_x23),
)


@pytest.mark.parametrize("x1", min_x1s)
@pytest.mark.parametrize("x2", min_x2s)
def test_minimum(x1, x2):
    """Minimum function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_y = ac_x1.minimum(ac_x2)
    torch_y = torch.minimum(torch_x1, torch_x2)
    _binary_function_verify(ac_x1, torch_x1, ac_x2, torch_x2, ac_y, torch_y)


@pytest.mark.parametrize("x1", min_x1s)
@pytest.mark.parametrize("x2", min_x2s)
def test_maximum(x1, x2):
    """Maximum function test"""
    ac_x1, torch_x1 = x1
    ac_x2, torch_x2 = x2
    ac_y = ac_x1.maximum(ac_x2)
    torch_y = torch.maximum(torch_x1, torch_x2)
    _binary_function_verify(ac_x1, torch_x1, ac_x2, torch_x2, ac_y, torch_y)
