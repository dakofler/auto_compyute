"""Tests for linear operations."""

import pytest
import torch.nn.functional as tF

import auto_compyute.nn.functional as F

from ...utils import close, get_random_floats, get_random_ints


def _emb_function_verify(w, torch_w, y, torch_y):
    assert close(y.data, torch_y)
    dy, torch_dy = get_random_floats(y.shape, False)
    y.backward(dy.data)
    torch_y.backward(torch_dy)
    assert close(w.grad, torch_w.grad, tol=1e-4)


ac_x1, torch_x1 = get_random_ints((16, 32), 0, 256)
ac_x2, torch_x2 = get_random_ints((8, 16, 32), 0, 256)
ac_w1, torch_w1 = get_random_floats((256, 64))
xs = ((ac_x1, torch_x1), (ac_x2, torch_x2))
ws = ((ac_w1, torch_w1),)


@pytest.mark.parametrize("x", xs)
@pytest.mark.parametrize("w", ws)
def test_embed(x, w):
    """Embedding function test"""
    ac_x, torch_x = x
    ac_w, torch_w = w
    ac_y = F.embedding(ac_x, ac_w)
    torch_y = tF.embedding(torch_x, torch_w)
    _emb_function_verify(ac_w, torch_w, ac_y, torch_y)
