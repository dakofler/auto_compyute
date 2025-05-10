"""Tests for activation function operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats
from tests.utils.test_factory import get_op_test
from tests.utils.verifications import verify_op

IN_SHAPES = ((10, 20), (10, 20, 30))
RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in IN_SHAPES)
LEAKY_RELU_ALPHAS = (0.1, 0.2)
SOFTMAX_DIMS = (0, 1)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
def test_gelu(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    ac_x, torch_x = x
    ac_y = ac.nn.functional.gelu(ac_x)
    torch_y = torch.nn.functional.gelu(torch_x, approximate="tanh")
    verify_op((ac_x,), ac_y, (torch_x,), torch_y)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
def test_relu(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("relu")((x,))


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("alpha", LEAKY_RELU_ALPHAS)
def test_leaky_relu(x: tuple[ac.Tensor, torch.Tensor], alpha: float) -> None:
    get_op_test("leaky_relu")((x,), alpha)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
def test_sigmoid(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("sigmoid")((x,))


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("dim", SOFTMAX_DIMS)
def test_softmax(x: tuple[ac.Tensor, torch.Tensor], dim: int) -> None:
    get_op_test("softmax")((x,), dim=dim)
