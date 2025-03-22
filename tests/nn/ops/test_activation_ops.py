"""Tests for activation function operations."""

import pytest
import torch.nn.functional as tF

import auto_compyute.nn.functional as F
from tests.utils import close, get_random_floats


def _unary_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad)


ac_x1, torch_x1 = get_random_floats((10, 20))
xs = ((ac_x1, torch_x1),)


@pytest.mark.parametrize("x", xs)
def test_gelu(x):
    """GELU function test"""
    ac_x, torch_x = x
    ac_y = F.gelu(ac_x)
    torch_y = tF.gelu(torch_x, approximate="tanh")
    _unary_function_verify(ac_x, torch_x, ac_y, torch_y)


@pytest.mark.parametrize("x", xs)
def test_relu(x):
    """ReLU function test"""
    ac_x, torch_x = x
    ac_y = F.relu(ac_x)
    torch_y = tF.relu(torch_x)
    _unary_function_verify(ac_x, torch_x, ac_y, torch_y)


alphas = (0.1, 0.2)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("alpha", alphas)
def test_leaky_relu(x, alpha):
    """Leaky ReLU function test"""
    ac_x, torch_x = x
    ac_y = F.leaky_relu(ac_x, alpha)
    torch_y = tF.leaky_relu(torch_x, alpha)
    _unary_function_verify(ac_x, torch_x, ac_y, torch_y)


@pytest.mark.parametrize("x", xs)
def test_sigmoid(x):
    """Sigmoid function test"""
    ac_x, torch_x = x
    ac_y = F.sigmoid(ac_x)
    torch_y = tF.sigmoid(torch_x)
    _unary_function_verify(ac_x, torch_x, ac_y, torch_y)


dims = (0, 1)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("dim", dims)
def test_softmax(x, dim):
    """Softmax function test"""
    ac_x, torch_x = x
    ac_y = F.softmax(ac_x, dim=dim)
    torch_y = tF.softmax(torch_x, dim)
    _unary_function_verify(ac_x, torch_x, ac_y, torch_y)
