"""Tests for attention operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.check import triple_input_op_check
from tests.utils.init import get_random_floats

IN_SHAPE = (8, 16, 32)
Q_RANDOM_FLOAT_TENSORS = (get_random_floats(IN_SHAPE),)
K_RANDOM_FLOAT_TENSORS = (get_random_floats(IN_SHAPE),)
V_RANDOM_FLOAT_TENSORS = (get_random_floats(IN_SHAPE),)
MASKS = (
    None,
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
    (ac_q, torch_q), (ac_k, torch_k), (ac_v, torch_v) = q, k, v
    (ac_mask, torch_mask) = mask if mask is not None else (None, None)
    ac_y = ac.nn.functional.scaled_dot_product_attention(ac_q, ac_k, ac_v, ac_mask)
    torch_y = torch.nn.functional.scaled_dot_product_attention(
        torch_q, torch_k, torch_v, torch_mask
    )
    triple_input_op_check(ac_q, ac_k, ac_v, ac_y, torch_q, torch_k, torch_v, torch_y)
