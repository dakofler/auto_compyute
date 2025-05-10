"""Tests for linear operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats
from tests.utils.test_factory import get_op_test

X_IN_SHAPES = ((16, 32), (8, 16, 32))
X_RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in X_IN_SHAPES)
W_RANDOM_FLOAT_TENSORS = (get_random_floats((64, 32)),)
B_RANDOM_FLOAT_TENSORS = (get_random_floats((64,)), (None, None))


@pytest.mark.parametrize("x", X_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("w", W_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("b", B_RANDOM_FLOAT_TENSORS)
def test_linear(
    x: tuple[ac.Tensor, torch.Tensor],
    w: tuple[ac.Tensor, torch.Tensor],
    b: tuple[ac.Tensor, torch.Tensor],
) -> None:
    get_op_test("linear")((x, w, b))
