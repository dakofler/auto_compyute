"""Tests for convolution operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats
from tests.utils.test_factory import get_op_test
from tests.utils.verifications import verify_op

PADDINGS = (0, 1)
STRIDES = (1, 2)
DILATIONS = (1, 2)
CONV1D_X_RANDOM_FLOAT_TENSORS = (get_random_floats((32, 16, 8)),)
CONV1D_W_RANDOM_FLOAT_TENSORS = (get_random_floats((4, 16, 3)),)
CONV1D_B_RANDOM_FLOAT_TENSORS = (get_random_floats((4,)),)
CONV2D_X_RANDOM_FLOAT_TENSORS = (get_random_floats((16, 3, 28, 28)),)
CONV2D_W_RANDOM_FLOAT_TENSORS = (get_random_floats((32, 3, 5, 5)),)
CONV2D_B_RANDOM_FLOAT_TENSORS = (get_random_floats((32,)),)
TCONV2D_X_RANDOM_FLOAT_TENSORS = (get_random_floats((16, 3, 14, 14)),)
TCONV2D_W_RANDOM_FLOAT_TENSORS = (get_random_floats((5, 3, 3, 3)),)
TCONV2D_B_RANDOM_FLOAT_TENSORS = (get_random_floats((5,)),)
TCONV2D_OUT_PADDINGS = (0, 1)
POOL_WINDOWS = (2, 3, 4)


@pytest.mark.parametrize("x", CONV1D_X_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("w", CONV1D_W_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("b", CONV1D_B_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("dilation", DILATIONS)
def test_conv1d(
    x: tuple[ac.Tensor, torch.Tensor],
    w: tuple[ac.Tensor, torch.Tensor],
    b: tuple[ac.Tensor, torch.Tensor],
    padding: int,
    stride: int,
    dilation: int,
) -> None:
    get_op_test("conv1d")((x, w, b), stride=stride, padding=padding, dilation=dilation)


@pytest.mark.parametrize("x", CONV2D_X_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("w", CONV2D_W_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("b", CONV2D_B_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("dilation", DILATIONS)
def test_conv2d(
    x: tuple[ac.Tensor, torch.Tensor],
    w: tuple[ac.Tensor, torch.Tensor],
    b: tuple[ac.Tensor, torch.Tensor],
    padding: int,
    stride: int,
    dilation: int,
) -> None:
    get_op_test("conv2d")((x, w, b), stride=stride, padding=padding, dilation=dilation)


@pytest.mark.parametrize("x", TCONV2D_X_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("w", TCONV2D_W_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("b", TCONV2D_B_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("dilation", DILATIONS)
@pytest.mark.parametrize("output_padding", TCONV2D_OUT_PADDINGS)
def test_conv_transpose2d(
    x: tuple[ac.Tensor, torch.Tensor],
    w: tuple[ac.Tensor, torch.Tensor],
    b: tuple[ac.Tensor, torch.Tensor],
    padding: int,
    stride: int,
    dilation: int,
    output_padding: int,
) -> None:
    if padding >= stride or padding >= dilation or output_padding > padding:
        return
    ac_x, torch_x = x
    ac_w, torch_w = w
    ac_b, torch_b = b
    ac_y = ac.nn.functional.conv_transpose2d(
        ac_x, ac_w, ac_b, stride, padding, output_padding, dilation
    )
    torch_y = torch.nn.functional.conv_transpose2d(
        torch_x,
        torch_w.transpose(0, 1),
        torch_b,
        stride,
        padding,
        output_padding,
        dilation=dilation,
    )
    verify_op((ac_x, ac_w, ac_b), ac_y, (torch_x, torch_w, torch_b), torch_y)


@pytest.mark.parametrize("x", CONV2D_X_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("window_size", POOL_WINDOWS)
def test_maxpool2d(x: tuple[ac.Tensor, torch.Tensor], window_size: int) -> None:
    get_op_test("maxpool2d", "max_pool2d")((x,), window_size)
