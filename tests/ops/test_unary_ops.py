"""Tests for unary operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats, get_random_positive_floats
from tests.utils.test_factory import get_unary_test_func

IN_SHAPES = ((10, 20), (10, 20, 30))
RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in IN_SHAPES)
RANDOM_POSITIVE_FLOAT_TENSORS = tuple(get_random_positive_floats(shape) for shape in IN_SHAPES)
POW_EXPONENTS = (2.0, 0.5, -0.5)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
def test_abs(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_unary_test_func("abs")(x)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
def test_exp(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_unary_test_func("exp")(x)


@pytest.mark.parametrize("x", RANDOM_POSITIVE_FLOAT_TENSORS)
@pytest.mark.parametrize("exponent", POW_EXPONENTS)
def test_pow(x: tuple[ac.Tensor, torch.Tensor], exponent: float) -> None:
    get_unary_test_func("pow")(x, exponent)


@pytest.mark.parametrize("x", RANDOM_POSITIVE_FLOAT_TENSORS)
def test_sqrt(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_unary_test_func("sqrt")(x)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
def test_tanh(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_unary_test_func("tanh")(x)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
def test_tril(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_unary_test_func("tril")(x)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
def test_triu(x: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_unary_test_func("triu")(x)
