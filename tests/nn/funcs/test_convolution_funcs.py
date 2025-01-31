"""Tests for convolution functions"""

import pytest
import torch.nn.functional as tF

import auto_compyute.nn.functional as F

from ...utils import close, get_data


def _conv_function_verify(x, torch_x, w, torch_w, b, torch_b, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_data(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad, tol=1e-3)
    assert close(w.grad, torch_w.grad, tol=1e-3)
    assert close(b.grad, torch_b.grad, tol=1e-3)


def _pool_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_data(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad, tol=1e-3)


ac_x1, torch_x1 = get_data((16, 3, 28, 28))
ac_w1, torch_w1 = get_data((32, 3, 5, 5))
ac_b1, torch_b1 = get_data((32,))
xs = ((ac_x1, torch_x1),)
ws = ((ac_w1, torch_w1),)
bs = ((ac_b1, torch_b1),)
paddings = (0, 2)
strides = (1, 2)
dilations = (1, 2)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("w", ws)
@pytest.mark.parametrize("b", bs)
@pytest.mark.parametrize("padding", paddings)
@pytest.mark.parametrize("stride", strides)
@pytest.mark.parametrize("dilation", dilations)
def test_conv(x, w, b, padding, stride, dilation):
    """Conv function test"""
    ac_x, torch_x = x
    ac_w, torch_w = w
    ac_b, torch_b = b
    ac_y = F.conv2d(ac_x, ac_w, ac_b, stride, padding, dilation)
    torch_y = tF.conv2d(torch_x, torch_w, torch_b, stride, padding, dilation)
    _conv_function_verify(ac_x, torch_x, ac_w, torch_w, ac_b, torch_b, ac_y, torch_y)


windows = (2, 3, 4)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("window_size", windows)
def test_maxpool2d(x, window_size):
    """Maxpool function test"""
    ac_x, torch_x = x
    ac_y = F.maxpool2d(ac_x, window_size)
    torch_y = tF.max_pool2d(torch_x, window_size)
    _pool_function_verify(ac_x, torch_x, ac_y, torch_y)
