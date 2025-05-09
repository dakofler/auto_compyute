"""Tests for attention operations."""

import pytest
import torch
import torch.nn.functional as tF

import auto_compyute as ac
import auto_compyute.nn.functional as F
from tests.utils.init import close, get_random_floats


def _attn_function_verify(q, torch_q, k, torch_k, v, torch_v, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(q.grad, torch_q.grad)
    assert close(k.grad, torch_k.grad)
    assert close(v.grad, torch_v.grad)


ac_q1, torch_q1 = get_random_floats((8, 16, 32))
ac_k2, torch_k2 = get_random_floats((8, 16, 32))
ac_v1, torch_v1 = get_random_floats((8, 16, 32))

qs = ((ac_q1, torch_q1),)
ks = ((ac_k2, torch_k2),)
vs = ((ac_v1, torch_v1),)
masks = (True, False)


@pytest.mark.parametrize("q", qs)
@pytest.mark.parametrize("k", ks)
@pytest.mark.parametrize("v", vs)
@pytest.mark.parametrize("mask", masks)
def test_attn(q, k, v, mask):
    """SDPA function test"""
    ac_q, torch_q = q
    ac_k, torch_k = k
    ac_v, torch_v = v

    if mask:
        seq_len = ac_q.shape[1]
        ac_mask = ac.full(seq_len, seq_len, value=float("-inf")).triu(1)
        torch_mask = torch.tensor(ac_mask.data)
    else:
        ac_mask = torch_mask = None
    ac_y = F.scaled_dot_product_attention(ac_q, ac_k, ac_v, ac_mask)
    torch_y = tF.scaled_dot_product_attention(torch_q, torch_k, torch_v, torch_mask)
    _attn_function_verify(ac_q, torch_q, ac_k, torch_k, ac_v, torch_v, ac_y, torch_y)
