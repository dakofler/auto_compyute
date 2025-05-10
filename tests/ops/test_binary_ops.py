"""Tests for binary operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats
from tests.utils.test_factory import get_op_test

IN_SHAPES_1 = ((10, 20),)
IN_SHAPES_2 = ((10, 20), (20,))
RANDOM_FLOAT_TENSORS_1 = tuple(get_random_floats(shape) for shape in IN_SHAPES_1)
RANDOM_FLOAT_TENSORS_2 = (*tuple(get_random_floats(shape) for shape in IN_SHAPES_2), (2.0, 2.0))
MATMUL_IN_SHAPES_2 = ((20, 5), (5, 20, 10))
MATMUL_RANDOM_FLOAT_TENSORS_2 = tuple(get_random_floats(shape) for shape in MATMUL_IN_SHAPES_2)
MIN_MAX_RANDOM_FLOAT_TENSORS_2 = (
    *tuple(get_random_floats(shape) for shape in IN_SHAPES_2),
    (0.0, torch.tensor(0.0)),
)


@pytest.mark.parametrize("x_1", RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", RANDOM_FLOAT_TENSORS_2)
def test_add(x_1: tuple[ac.Tensor, torch.Tensor], x_2: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("__add__")((x_1, x_2))


@pytest.mark.parametrize("x_1", RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", RANDOM_FLOAT_TENSORS_2)
def test_sub(x_1: tuple[ac.Tensor, torch.Tensor], x_2: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("__sub__")((x_1, x_2))


@pytest.mark.parametrize("x_1", RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", RANDOM_FLOAT_TENSORS_2)
def test_mul(x_1: tuple[ac.Tensor, torch.Tensor], x_2: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("__mul__")((x_1, x_2))


@pytest.mark.parametrize("x_1", RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", RANDOM_FLOAT_TENSORS_2)
def test_truediv(x_1: tuple[ac.Tensor, torch.Tensor], x_2: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("__truediv__")((x_1, x_2))


@pytest.mark.parametrize("x_1", RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", MATMUL_RANDOM_FLOAT_TENSORS_2)
def test_matmul(x_1: tuple[ac.Tensor, torch.Tensor], x_2: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("__matmul__")((x_1, x_2))


@pytest.mark.parametrize("x_1", RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", MIN_MAX_RANDOM_FLOAT_TENSORS_2)
def test_maximum(x_1: tuple[ac.Tensor, torch.Tensor], x_2: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("maximum")((x_1, x_2))


@pytest.mark.parametrize("x_1", RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", MIN_MAX_RANDOM_FLOAT_TENSORS_2)
def test_minimum(x_1: tuple[ac.Tensor, torch.Tensor], x_2: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("minimum")((x_1, x_2))
