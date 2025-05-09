"""Tests for convolution operations."""

import pytest
import torch.nn.functional as tF

import auto_compyute.nn.functional as F
from tests.utils.init import close, get_random_floats


def _conv_function_verify(x, torch_x, w, torch_w, b, torch_b, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad, tol=1e-3)
    assert close(w.grad, torch_w.grad, tol=1e-3)
    assert close(b.grad, torch_b.grad, tol=1e-3)


def _pool_function_verify(x, torch_x, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(x.grad, torch_x.grad, tol=1e-3)


c1d_ac_x1, c1d_torch_x1 = get_random_floats((32, 16, 8))  # B, I, T
c1d_ac_w1, c1d_torch_w1 = get_random_floats((4, 16, 3))  # O, I, K
c1d_ac_b1, c1d_torch_b1 = get_random_floats((4,))
c1d_xs = ((c1d_ac_x1, c1d_torch_x1),)
c1d_ws = ((c1d_ac_w1, c1d_torch_w1),)
c1d_bs = ((c1d_ac_b1, c1d_torch_b1),)
paddings = (0, 1)
strides = (1, 2)
dilations = (1, 2)


@pytest.mark.parametrize("x", c1d_xs)
@pytest.mark.parametrize("w", c1d_ws)
@pytest.mark.parametrize("b", c1d_bs)
@pytest.mark.parametrize("padding", paddings)
@pytest.mark.parametrize("stride", strides)
@pytest.mark.parametrize("dilation", dilations)
def test_conv1d(x, w, b, padding, stride, dilation):
    """Conv 1d function test"""
    ac_x, torch_x = x
    ac_w, torch_w = w
    ac_b, torch_b = b
    ac_y = F.conv1d(ac_x, ac_w, ac_b, stride, padding, dilation)
    torch_y = tF.conv1d(torch_x, torch_w, torch_b, stride, padding, dilation)
    _conv_function_verify(ac_x, torch_x, ac_w, torch_w, ac_b, torch_b, ac_y, torch_y)


c2d_ac_x1, c2d_torch_x1 = get_random_floats((16, 3, 28, 28))
c2d_ac_w1, c2d_torch_w1 = get_random_floats((32, 3, 5, 5))
c2d_ac_b1, c2d_torch_b1 = get_random_floats((32,))
c2d_bs = ((c2d_ac_b1, c2d_torch_b1),)
c2d_xs = ((c2d_ac_x1, c2d_torch_x1),)
c2d_ws = ((c2d_ac_w1, c2d_torch_w1),)


@pytest.mark.parametrize("x", c2d_xs)
@pytest.mark.parametrize("w", c2d_ws)
@pytest.mark.parametrize("b", c2d_bs)
@pytest.mark.parametrize("padding", paddings)
@pytest.mark.parametrize("stride", strides)
@pytest.mark.parametrize("dilation", dilations)
def test_conv2d(x, w, b, padding, stride, dilation):
    """Conv 2d function test"""
    ac_x, torch_x = x
    ac_w, torch_w = w
    ac_b, torch_b = b
    ac_y = F.conv2d(ac_x, ac_w, ac_b, stride, padding, dilation)
    torch_y = tF.conv2d(torch_x, torch_w, torch_b, stride, padding, dilation)
    _conv_function_verify(ac_x, torch_x, ac_w, torch_w, ac_b, torch_b, ac_y, torch_y)


t2d_ac_x1, t2d_torch_x1 = get_random_floats((16, 3, 14, 14))
t2d_ac_w1, t2d_torch_w1 = get_random_floats((5, 3, 3, 3))
t2d_ac_b1, t2d_torch_b1 = get_random_floats((5,))
t2d_xs = ((t2d_ac_x1, t2d_torch_x1),)
t2d_ws = ((t2d_ac_w1, t2d_torch_w1),)
t2d_bs = ((t2d_ac_b1, t2d_torch_b1),)
t2d_strides = (1, 2)
t2d_paddings = (0, 1)
t2d_output_paddings = (0, 1)
t2d_dilations = (1, 2)


@pytest.mark.parametrize("x", t2d_xs)
@pytest.mark.parametrize("w", t2d_ws)
@pytest.mark.parametrize("b", t2d_bs)
@pytest.mark.parametrize("stride", t2d_strides)
@pytest.mark.parametrize("padding", t2d_paddings)
@pytest.mark.parametrize("output_padding", t2d_output_paddings)
@pytest.mark.parametrize("dilation", t2d_dilations)
def test_conv_transpose2d(x, w, b, stride, padding, output_padding, dilation):
    """Conv function test"""
    if padding >= stride or padding >= dilation or output_padding > padding:
        return
    ac_x, torch_x = x
    ac_w, torch_w = w
    ac_b, torch_b = b
    ac_y = F.conv_transpose2d(ac_x, ac_w, ac_b, stride, padding, output_padding, dilation)
    torch_y = tF.conv_transpose2d(
        torch_x,
        torch_w.transpose(0, 1),
        torch_b,
        stride,
        padding,
        output_padding,
        dilation=dilation,
    )
    _conv_function_verify(ac_x, torch_x, ac_w, torch_w, ac_b, torch_b, ac_y, torch_y)


windows = (2, 3, 4)


@pytest.mark.parametrize("x", c2d_xs)
@pytest.mark.parametrize("window_size", windows)
def test_maxpool2d(x, window_size):
    """Maxpool function test"""
    ac_x, torch_x = x
    ac_y = F.maxpool2d(ac_x, window_size)
    torch_y = tF.max_pool2d(torch_x, window_size)
    _pool_function_verify(ac_x, torch_x, ac_y, torch_y)
