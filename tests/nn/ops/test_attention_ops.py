"""Tests for attention operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats
from tests.utils.test_factory import get_op_test

IN_SHAPE = (8, 16, 32)
Q_RANDOM_FLOAT_TENSORS = (get_random_floats(IN_SHAPE),)
K_RANDOM_FLOAT_TENSORS = (get_random_floats(IN_SHAPE),)
V_RANDOM_FLOAT_TENSORS = (get_random_floats(IN_SHAPE),)
MASKS = (
    (None, None),
    (
        ac_mask := ac.full(IN_SHAPE[1], IN_SHAPE[1], value=float("-inf")).triu(1),
        torch.tensor(ac_mask.data),
    ),
)


@pytest.mark.parametrize("q", Q_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("k", K_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("v", V_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("mask", MASKS)
def test_attn(
    q: tuple[ac.Tensor, torch.Tensor],
    k: tuple[ac.Tensor, torch.Tensor],
    v: tuple[ac.Tensor, torch.Tensor],
    mask: tuple[ac.Tensor, torch.Tensor] | None,
) -> None:
    get_op_test("scaled_dot_product_attention")((q, k, v, mask))
