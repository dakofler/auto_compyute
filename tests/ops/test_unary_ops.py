"""Tests for unary operations."""

from typing import Any, Callable

import pytest
import torch

import auto_compyute as ac
from tests.utils.check import single_input_op_check
from tests.utils.init import get_random_floats, get_random_positive_floats

SHAPES = ((10, 20), (10, 20, 30))
FLOAT_IN_TENSORS = tuple(get_random_floats(shape) for shape in SHAPES)
POSITIVE_FLOAT_IN_TENSORS = tuple(get_random_positive_floats(shape) for shape in SHAPES)


def _get_unary_test_func(func_name: str) -> Callable[[Any], None]:
    def _test(in_tensors: tuple[ac.Tensor, torch.Tensor]) -> None:
        ac_in_tensor, torch_in_tensor = in_tensors
        ac_out_tensor = getattr(ac_in_tensor, func_name)()
        torch_out_tensor = getattr(torch, func_name)(torch_in_tensor)
        single_input_op_check(ac_in_tensor, ac_out_tensor, torch_in_tensor, torch_out_tensor)

    return _test


@pytest.mark.parametrize("in_tensors", FLOAT_IN_TENSORS)
def test_abs(in_tensors: tuple[ac.Tensor, torch.Tensor]) -> None:
    _get_unary_test_func("abs")(in_tensors)


@pytest.mark.parametrize("in_tensors", FLOAT_IN_TENSORS)
def test_exp(in_tensors: tuple[ac.Tensor, torch.Tensor]) -> None:
    _get_unary_test_func("exp")(in_tensors)


@pytest.mark.parametrize("in_tensors", POSITIVE_FLOAT_IN_TENSORS)
def test_sqrt(in_tensors: tuple[ac.Tensor, torch.Tensor]) -> None:
    _get_unary_test_func("sqrt")(in_tensors)


@pytest.mark.parametrize("in_tensors", FLOAT_IN_TENSORS)
def test_tanh(in_tensors: tuple[ac.Tensor, torch.Tensor]) -> None:
    _get_unary_test_func("tanh")(in_tensors)


@pytest.mark.parametrize("in_tensors", FLOAT_IN_TENSORS)
def test_tril(in_tensors: tuple[ac.Tensor, torch.Tensor]) -> None:
    _get_unary_test_func("tril")(in_tensors)


@pytest.mark.parametrize("in_tensors", FLOAT_IN_TENSORS)
def test_triu(in_tensors: tuple[ac.Tensor, torch.Tensor]) -> None:
    _get_unary_test_func("triu")(in_tensors)
