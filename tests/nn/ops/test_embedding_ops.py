"""Tests for embedding operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats, get_random_ints
from tests.utils.test_factory import get_op_test

X_IN_SHAPES = ((16, 32), (8, 16, 32))
X_RANDOM_INT_TENSORS = tuple(get_random_ints(shape, 0, 256) for shape in X_IN_SHAPES)
W_RANDOM_FLOAT_TENSORS = (get_random_floats((256, 64)),)


@pytest.mark.parametrize("x", X_RANDOM_INT_TENSORS)
@pytest.mark.parametrize("w", W_RANDOM_FLOAT_TENSORS)
def test_embed(x: tuple[ac.Tensor, torch.Tensor], w: tuple[ac.Tensor, torch.Tensor]) -> None:
    get_op_test("embedding")((x, w))
